// Cache licensed Wikimedia crop photographs for same-origin delivery where network access permits.
// Network failures never block the frontend build: the browser has remote URL fallbacks.
import {readFile,mkdir,writeFile,stat} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const photos=JSON.parse(await readFile(path.join(root,'src/photos.json'),'utf8'));
await mkdir(path.join(root,'public/photos'),{recursive:true});
await Promise.all(Object.entries(photos).map(async ([crop,p])=>{
 const destination=path.join(root,'public',p.local.replace(/^\//,''));
 try{if((await stat(destination)).size>1024)return;}catch{}
 for(const url of [p.url,p.original]){
  try{
   const r=await fetch(url,{signal:AbortSignal.timeout(8000),headers:{'User-Agent':'AfghanistanCropAI/3.1 (educational crop-photo cache; Wikimedia Commons credits in app)'}});
   if(!r.ok||!r.headers.get('content-type')?.startsWith('image/'))continue;
   if(Number(r.headers.get('content-length'))>8000000)continue;
   const b=Buffer.from(await r.arrayBuffer()); if(b.length<1024||b.length>8000000)continue;
   await writeFile(destination,b);console.log('Cached photo:',crop);return;
  }catch{}
 }
 console.warn('Photo cache unavailable:',crop,'— using browser URL fallback');
}));
