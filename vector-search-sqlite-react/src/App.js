import './App.css';
import React from 'react';
import { tableFromIPC } from 'apache-arrow';
import codebk from './codewords.json';
import { pipeline, env } from '@xenova/transformers';
import RangeSlider from 'react-bootstrap-range-slider';
import {format} from 'd3-format';
import { distTopK, queryToTiledDist, makeONNXRunnables, flattenCodebook } from 'embeddingdb.js';

const format3SI = format('.3s')

// const pqD = 384;
// const pqM = 48;
// const pqDs = 8;
// const pqK = 128;

const codebkflat = flattenCodebook(codebk);

const numpyChunkSize = 200000;

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {query: "where a word means like how it sounds", firstLetter: "", chunkCount: 10, k: 10, embeddings: [], dists: [], firstLetters: []};
    env.localModelPath = './models/' ;
  }
  async componentDidMount() {
    let extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    this.setState({extractor});
    const {filteredtopkinf, pqdistinf} = await makeONNXRunnables();
    this.setState({filteredtopkinf, pqdistinf}, () => this.loadEmbeddings(this.state.chunkCount));
  }
  async loadEmbeddings(maxCount) {
    const embeddings = this.state.embeddings.slice(0, maxCount);
    const firstLetters = Float32Array.from({length: maxCount * numpyChunkSize});
    const dists = Float32Array.from({length: firstLetters.length}, () => 1234567890);
    const maxoldlen = Math.min(firstLetters.length, this.state.firstLetters.length);
    for(let i = 0; i < maxoldlen; i++) {
      firstLetters[i] = this.state.firstLetters[i];
      dists[i] = this.state.dists[i];
    }
    this.setState({embeddings, firstLetters, dists, loadingEmbeddings: true});
    for(let i = embeddings.length; i < maxCount; i++) {
      const embeddingShardPath = `./embedding-${i}-shardsize-${numpyChunkSize}.arrow`;
      const titleShardPath = `./title-${i}-shardsize-${numpyChunkSize}.arrow`;
      const eResponse = await fetch(embeddingShardPath);
      const tResponse = await fetch(titleShardPath);
      const eBuffer = await eResponse.arrayBuffer();
      const tBuffer = await tResponse.arrayBuffer();
      const eArrow = tableFromIPC(eBuffer);
      const title = tableFromIPC(tBuffer);
      const data = eArrow.data[0].children[0].values;
      const lastEmbedding = {
        data,
        offset: i * numpyChunkSize,
        title,
      };
      embeddings.push(lastEmbedding);
      for(let j = 0; j < lastEmbedding.data.length / codebk.length; j++) {
        firstLetters[lastEmbedding.offset + j] = lastEmbedding.title.get(j)['title'].charCodeAt(0);
      }
      await this.makeQuery({onlyLast: true, skipEmbed: true});
    }
    this.setState({loadingEmbeddings: false}, () => this.makeQuery({onlyFilter: true}))
  }
  async makeQuery({onlyLast, onlyFilter, skipEmbed}) {
    const {extractor, embeddings, query, queryEmbedding, firstLetters, firstLetter, filteredtopkinf, pqdistinf, dists, k, } = this.state;
    const firstLetterInt = firstLetter.length === 0 ? 0 : firstLetter.charCodeAt(0);
    if(onlyFilter) {
      const startTime = Date.now();
      const topk = await distTopK(filteredtopkinf, dists, firstLetters, firstLetterInt, 0, 1024, k);
      const endTime = Date.now();
      this.setState({topk, distTime: `${endTime - startTime}`, minilmtime: "—"});
      return;
    }
    if(!skipEmbed || !queryEmbedding){
      const minilmstart = Date.now();
      const minilmresult = await extractor(query, {pooling: "mean", normalize: true});
      const minilmend = Date.now();
      this.setState({minilmtime: minilmend - minilmstart, queryEmbedding: minilmresult.data}, () => this.makeQuery({skipEmbed: true}));
      return;
    }
    const embeddingCounter = onlyLast ? (embeddings.length - 1) : 0;
    await queryToTiledDist(queryEmbedding, embeddings, codebk, codebkflat, pqdistinf, dists, firstLetters, firstLetterInt, filteredtopkinf, k, (h) => this.setState(h), () => query === this.state.query, embeddingCounter);
  }
  canPerformSearch () {
    const {extractor, embeddings} = this.state;
    return !!extractor && !!embeddings;
  }
  render() {
    const {query, minilmtime, distTime, firstLetter, topk, embeddings, extractor, k, newEmbeddingSliderValue} = this.state;
    return (<div className="App">
      <h1>Wikipedia search-by-vibes</h1>
      <h2>
        <textarea value={query} placeholder="query to make" onChange={e => this.setState({query: e.target.value}, () => this.canPerformSearch() ? this.makeQuery({}) : null)} disabled={!this.canPerformSearch()} ></textarea>
        <br/>
        <input type="text" value={firstLetter} placeholder="first letter to filter on"
             disabled={!this.canPerformSearch()}
             onChange={e => this.setState({firstLetter: e.target.value.slice(0,1)}, () => this.makeQuery({onlyFilter: true}))}>
        </input>
        <RangeSlider tooltip='on' tooltipLabel={currentValue => currentValue === 1 ? 'TOP RESULT' : `TOP ${currentValue} RESULTS`} min={1} max={200} value={k} tooltipPlacement={'top'} onChange={e => this.setState({k: e.target.value}, () => this.makeQuery({onlyFilter: true}))} disabled={!this.canPerformSearch()} />
        </h2>
      <div className='topk-results'>
        {
          (!topk) ? (
            (embeddings ? "" : "Waiting for embeddings to load. ") + (extractor ? "" : " Waiting for semantic sentence language model to load. " ) + (" Waiting to run query.")
          ) : [...Int32Array.from(topk, e => Number(e))].filter(idx => idx < embeddings.length * numpyChunkSize).map((idx) => (
          <div className='topk-result' key={`topk${idx}`}>
            {(() => {
              let title = (embeddings[Math.floor(idx / numpyChunkSize)].title).get(idx % numpyChunkSize)['title'];
              return <span className='topk-result-title'>
                <a href={"https://en.wikipedia.org/wiki/" + encodeURIComponent(title.replaceAll(" ", "_"))}>{title}</a>
              </span>
            })()}
            <span className='topk-result-rank' title='the rank of the compressed size of the page, 1 is the largest page on Wikipedia'>{format3SI(idx).toUpperCase()}</span>
          </div>))
        }
      </div>
      <h4>minilm: {minilmtime} ms <br/> topk: {distTime} ms</h4>
      <h2>
        <RangeSlider tooltip='on' tooltipLabel={currentValue => (this.state.loadingEmbeddings ? "⏳ " : "") + (currentValue === embeddings.length ? `SEARCHING ${embeddings.length * numpyChunkSize / 1000000} MILLION PAGES OFFLINE.` : `LOAD ${currentValue * numpyChunkSize / 1000000} MILLION PAGES`)}
          min={0} max={32} value={newEmbeddingSliderValue || embeddings.length}
          tooltipPlacement='top'
          disabled={!this.canPerformSearch()}
          onChange={e => this.state.loadingEmbeddings ? "" : this.setState({newEmbeddingSliderValue: e.target.value})}
          onAfterChange={e => {if(this.state.loadingEmbeddings) { return; } const newVal = e.target.value; this.setState({newEmbeddingSliderValue: undefined, loadingEmbeddings: true}, () => this.loadEmbeddings(newVal))}}
          />
      </h2>
      <h3 style={ {textAlign: 'right'}} >
         —Lee Butterman, June 2023.<br/><a href="https://www.leebutterman.com/2023/06/01/offline-realtime-embedding-search.html">how this was made ⋙</a>
      </h3>

    </div>);
  }
}

export default App;
