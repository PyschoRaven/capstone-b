/* Transitions for existing tab handlers; normal links use cross-document CSS transitions. */
(() => {
 let replaying=false;
 const reduced=()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches;
 function activeContent(){return document.querySelector('.ag-view:not([hidden])')||document.querySelector('body[data-analytics-panel="security"] #security-review')||document.querySelector('.sys-app')||document.querySelector('.gt-app')}
 function replay(target){replaying=true;try{target.click()}finally{replaying=false}}
 document.addEventListener('click',event=>{
  if(replaying||event.defaultPrevented||event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey||reduced())return;
  const target=event.target.closest('[data-tab],[data-tab-link],[data-literal-view],[data-analytics-view],[data-system-panel]');
  if(!target||target.disabled||target.getAttribute('aria-pressed')==='true'||target.classList.contains('is-active'))return;
  if(typeof document.startViewTransition==='function'){
   event.preventDefault();event.stopImmediatePropagation();
   try{document.startViewTransition(()=>replay(target))}catch{replay(target)}
  }else{
   requestAnimationFrame(()=>{const panel=activeContent();if(panel&&typeof panel.animate==='function')panel.animate([{opacity:0,transform:'translateX(12px)'},{opacity:1,transform:'translateX(0)'}],{duration:240,easing:'ease-out'})})
  }
 },true);
 // Browsers without native view transitions still get a brief page entrance.
 if(typeof document.startViewTransition!=='function'&&!reduced()){
  const panel=document.querySelector('.ag-app,.gt-app');
  if(panel&&typeof panel.animate==='function')panel.animate([{opacity:0,transform:'translateX(12px)'},{opacity:1,transform:'translateX(0)'}],{duration:240,easing:'ease-out'});
 }
})();
