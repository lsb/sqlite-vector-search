import './App.css';
import React from 'react';
import { tableFromIPC } from 'apache-arrow';
import codebk from './codewords.json';
import { pipeline, env } from '@xenova/transformers';
import * as ort from 'onnxruntime-web';
import RangeSlider from 'react-bootstrap-range-slider';
import millify from 'millify';
import {filteredTopKAsc, pqDist} from 'pq.js';

const InferenceSession = ort.InferenceSession;
const Tensor = ort.Tensor;

const codebookshape = [codebk.length, codebk[0].length, codebk[0][0].length];
// const pqD = 384;
// const pqM = 48;
// const pqDs = 8;
// const pqK = 128;
const codebkflat = Float32Array.from({length: codebk.length * codebk[0].length * codebk[0][0].length}, (e,i) => codebk[Math.floor(i / codebk[0].length / codebk[0][0].length)][Math.floor(i / codebk[0][0].length) % codebk[0].length][i % codebk[0][0].length])
const codebkT = new Tensor("float32", codebkflat, codebookshape)

const numpyChunkSize = 200000;

async function distTopK(inferenceSession, dists, filterColumn, filterValue, filterZero, filterShim, k) {
  // console.log("trying to distTopk");
  const {output: {data: topk}} = await inferenceSession.run({
    "input": (new Tensor("float32", dists)),
    "filterColumn": (new Tensor("float32", filterColumn)),
    "filterValue": (new Tensor("float32", [filterValue])),
    "filterZero": (new Tensor("float32", [filterZero])),
    "filterShim": (new Tensor("float32", [filterShim])),
    "k": (new Tensor("uint8", [k])),
  });
  return topk;
}

async function queryDist(inferenceSession, query, codebook, codebookShape, embeddings, embeddingTensorShape) {
  // console.log("trying to querydist", {query, codebook, codebookShape, embeddings, embeddingTensorShape});
  const {output: {data: distTile}} = await inferenceSession.run({
    "query": (new Tensor("float32", query)),
    "codebook": (new Tensor("float32", codebook, codebookShape)),
    "embeddings": (new Tensor("uint8", embeddings, embeddingTensorShape)),

  })
  return distTile;
}

async function queryToTiledDist(query, embeddings, pqdistinf, dists, firstLetters, firstLetterInt, filteredtopkinf, k, intermediateValueFn, continueFn, embeddingCounter=0) {
  // compute distances a tile at a time, update in the dists array, compute topk not more frequently than every 30 ms (to avoid excessive screen repainting)
  // on an iphone, distances for 1M embeddings runs in ~100 ms whereas topk for 1M floats runs in ~3ms
  // call a sentinel function to halt processing if the outside state changes (there is probably a better way)
  const chunkSize = 100000;
  let lastPaint = Date.now();
  const maxTick = 30;
  const timingStrings = [];
  // console.log({dists: dists.length, chunkSize, firstLetters: firstLetters.length});
  for(; embeddingCounter<embeddings.length; embeddingCounter++){
    // console.log(`starting embedding ${embeddingCounter}`)
    const {data: embeddingData, offset: embeddingOffset} = embeddings[embeddingCounter];
    for(let i=0; i < (embeddingData.length / codebk.length) && continueFn(); i+=chunkSize) {
      // console.log({i, embeddingDatalength: embeddingData.length, continuefn: continueFn()})
      const startTime = Date.now();
      const startEmbeddingPosition = i * codebk.length;
      const embeddingTileLength = chunkSize * codebk.length;
      const embeddingTensorShape = [chunkSize, codebk.length];
      const embeddingTile = new Uint8Array(embeddingData.buffer, startEmbeddingPosition + embeddingData.byteOffset, embeddingTileLength);
      // console.log({i, startEmbeddingPosition, embeddingTileLength, embeddingTensorShape, workingShape: [embeddings.length / codebk.length, codebk.length], embeddingTile, embeddings});
      const distTile = await queryDist(pqdistinf, query, codebkflat, codebookshape, embeddingTile, embeddingTensorShape);
      // console.log("got dists")
      for(let j=0;j<chunkSize;j++) {
        dists[embeddingOffset+i+j] = distTile[j];
      }
      // console.log("wrote dists");
      const distTime = Date.now();
      timingStrings.push(`${distTime - startTime}`)
      // console.log("2")
      intermediateValueFn({dists, distTime: timingStrings.join(), lastPaint});
      // console.log("1")
      if((i === 0 && embeddingCounter === 0) || (distTime - lastPaint > maxTick)) {
        // console.log("0, or we're over time")
        const topk = await distTopK(filteredtopkinf, dists, firstLetters, firstLetterInt, 0, 1024, k);
        const topktime = Date.now();
        timingStrings.push(`-${topktime-distTime} `);
        intermediateValueFn({topk});
        await (new Promise(r => setTimeout(r,0)));
        lastPaint = Date.now()
      }
    }
  }
  if(continueFn()) {
    const topk = await distTopK(filteredtopkinf, dists, firstLetters, firstLetterInt, 0, 1024, k);
    intermediateValueFn({dists, distTime: timingStrings.join(), topk});
  }
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {query: "where a word means like how it sounds", firstLetter: "", chunkCount: 10, k: 10, embeddings: [], dists: [], firstLetters: []};
    env.localModelPath = './models/'
  }
  async componentDidMount() {
    let extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    this.setState({extractor});
    const filteredtopkinf = await InferenceSession.create(filteredTopKAsc);
    const pqdistinf = await InferenceSession.create(pqDist, {executionProviders: ['wasm']});
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
    const responses = {};
    for(let i = embeddings.length; i < maxCount; i++) {
      const embeddingShardPath = `./embedding-${i}-shardsize-${numpyChunkSize}.arrow`;
      const titleShardPath = `./title-${i}-shardsize-${numpyChunkSize}.arrow`;
      const eResponse = await fetch(embeddingShardPath);
      const tResponse = await fetch(titleShardPath);
      responses[i] = {eResponse, tResponse};
    }
    for(let i = embeddings.length; i < maxCount; i++) {
      const {eResponse, tResponse} = responses[i];
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
    await queryToTiledDist(queryEmbedding, embeddings, pqdistinf, dists, firstLetters, firstLetterInt, filteredtopkinf, k, (h) => this.setState(h), () => query === this.state.query, embeddingCounter);
  }
  render() {
    const {query, minilmtime, distTime, firstLetter, filteredtopkinf, topk, embeddings, extractor, k, newEmbeddingSliderValue} = this.state;
    if(!extractor) {
      return (<div>Waiting for MiniLM to load</div>);
    }
    if(!filteredtopkinf) {
      return (<div>Waiting for WASM to load</div>)
    }
    if(!embeddings) {
      return (<div>Waiting for the first embedding to load</div>)
    }
    return (<div className="App">
      <h1>Wikipedia search-by-vibes</h1>
      <h2>
        <textarea value={query} placeholder="query to make" onChange={e => this.setState({query: e.target.value}, () => this.makeQuery({}))}></textarea>
        <br/>
        <input type="text" value={firstLetter} placeholder="first letter to filter on"
             onChange={e => this.setState({firstLetter: e.target.value.slice(0,1)}, () => this.makeQuery({onlyFilter: true}))}>
        </input>
        <RangeSlider tooltip='on' tooltipLabel={currentValue => currentValue === 1 ? 'TOP RESULT' : `TOP ${currentValue} RESULTS`} min={1} max={200} value={k} tooltipPlacement={'top'} onChange={e => this.setState({k: e.target.value}, () => this.makeQuery({onlyFilter: true}))} />
        </h2>
      <div className='topk-results'>
        {
          (!topk) ? "Waiting for topk to run once" : [...Int32Array.from(topk, e => Number(e))].filter(idx => idx < embeddings.length * numpyChunkSize).map((idx) => (
          <div className='topk-result' key={`topk${idx}`}>
            <span className='topk-result-title'>{(embeddings[Math.floor(idx / numpyChunkSize)].title).get(idx % numpyChunkSize)['title']}</span>
            <span className='topk-result-rank' title='the rank of the compressed size of the page, 1 is the largest page on Wikipedia'>{millify(idx, {lowercase: true, precision: 0})}</span>
            </div>))
        }
      </div>
      <h4>minilm: {minilmtime} ms <br/> topk: {distTime} ms</h4>
      <h2>
        <RangeSlider tooltip='on' tooltipLabel={currentValue => (this.state.loadingEmbeddings ? "⏳ " : "") + (currentValue === embeddings.length ? `SEARCHING ${embeddings.length * numpyChunkSize / 1000000} MILLION PAGES OFFLINE.` : `LOAD ${currentValue * numpyChunkSize / 1000000} MILLION PAGES`)}
          min={0} max={32} value={newEmbeddingSliderValue || embeddings.length}
          tooltipPlacement='top'
          onChange={e => this.state.loadingEmbeddings ? "" : this.setState({newEmbeddingSliderValue: e.target.value})}
          onAfterChange={e => {if(this.state.loadingEmbeddings) { return; } const newVal = e.target.value; this.setState({newEmbeddingSliderValue: undefined, loadingEmbeddings: true}, () => this.loadEmbeddings(newVal))}}
          />
      </h2>
      <h3 style={ {textAlign: 'right'}} >
         —Lee Butterman, June 2023.<br/><a href="https://leebutterman.com">how this was made ⋙</a>
      </h3>

    </div>);
  }
}

export default App;
