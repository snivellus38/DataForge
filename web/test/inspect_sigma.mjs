import { readFileSync } from "node:fs";
import { BDHModel, equivalenceCheck } from "../src/bdh.js";
const man = JSON.parse(readFileSync("web/public/model.json","utf8"));
const bin = readFileSync("web/public/model.bin");
const m = new BDHModel(man, bin.buffer.slice(bin.byteOffset, bin.byteOffset+bin.byteLength));
const pre = JSON.parse(readFileSync("web/public/presets.json","utf8"));
const toks = pre.hook.bytes;
const r = equivalenceCheck(m, toks, {});
const { sigma, N, D, T } = r;
let nz=0, mx=0, mn=0, sum=0;
const rowNorm = new Float64Array(N);
for (let n=0;n<N;n++){ let s=0; for(let d=0;d<D;d++){const v=sigma[n*D+d]; s+=v*v; if(v!==0)nz++; mx=Math.max(mx,v); mn=Math.min(mn,v); sum+=Math.abs(v);} rowNorm[n]=Math.sqrt(s); }
const active = rowNorm.filter(v=>v>1e-6).length;
const sorted=[...rowNorm].sort((a,b)=>b-a);
console.log(`T=${T} tokens, sigma ${N}x${D}`);
console.log(`nonzero cells: ${nz}/${N*D} (${(100*nz/(N*D)).toFixed(1)}%)`);
console.log(`active rows (||row||>1e-6): ${active}/${N} (${(100*active/N).toFixed(1)}%)`);
console.log(`range: [${mn.toFixed(3)}, ${mx.toFixed(3)}]  mean|v|=${(sum/(N*D)).toFixed(4)}`);
console.log(`row-norm percentiles: max=${sorted[0].toFixed(2)} p50=${sorted[Math.floor(N/2)].toFixed(3)} p90=${sorted[Math.floor(N*0.9)].toFixed(4)}`);
// how concentrated? top-k rows carrying 90% of energy
let tot=sorted.reduce((a,b)=>a+b*b,0), acc=0, k=0;
while(acc<0.9*tot && k<N){acc+=sorted[k]*sorted[k];k++;}
console.log(`rows carrying 90% of energy: ${k}/${N}`);
// sparsity of x_sparse (the "5% active neurons" analogue)
const f=m.forward(toks);
let xnz=0; for(let i=0;i<f.xSparse.length;i++) if(f.xSparse[i]>0) xnz++;
console.log(`x_sparse active: ${(100*xnz/f.xSparse.length).toFixed(1)}% of ${f.xSparse.length} entries`);
