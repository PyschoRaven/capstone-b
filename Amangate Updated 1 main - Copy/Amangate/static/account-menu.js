// Native popovers place account menus above every page stacking context.
(() => {
 document.querySelectorAll('details.literal-menu').forEach((details,index)=>{
  const trigger=details.querySelector('summary'),menu=details.querySelector(':scope>div');if(!trigger||!menu)return;
  document.body.append(menu);
  menu.classList.add('account-popover');menu.setAttribute('popover','manual');menu.id='account-popover-'+index;
  trigger.setAttribute('aria-controls',menu.id);trigger.setAttribute('aria-expanded','false');
  let trackingFrame=0;
  function place(){
   const box=trigger.getBoundingClientRect();
   if(box.bottom<=0||box.top>=innerHeight||box.right<=0||box.left>=innerWidth){close();return}
   const width=Math.min(240,innerWidth-24);menu.style.setProperty('width',width+'px');
   const height=menu.getBoundingClientRect().height;
   const below=box.bottom+10;
   // Flip above when needed, always keeping the menu attached to the avatar.
   const top=below+height<=innerHeight-12?below:box.top-height-10>=12?box.top-height-10:below;
   menu.style.setProperty('left',Math.max(12,Math.min(innerWidth-width-12,box.right-width))+'px','important');
   menu.style.setProperty('top',top+'px','important');
  }
  function follow(){if(!details.open||!menu.matches(':popover-open')){trackingFrame=0;return}place();if(details.open)trackingFrame=requestAnimationFrame(follow)}
  function close(){cancelAnimationFrame(trackingFrame);trackingFrame=0;if(menu.matches(':popover-open'))menu.hidePopover();details.open=false;trigger.setAttribute('aria-expanded','false')}
  trigger.addEventListener('click',e=>{e.preventDefault();if(details.open){close();return}document.querySelectorAll('details.literal-menu[open]').forEach(d=>d.open=false);details.open=true;menu.showPopover();trigger.setAttribute('aria-expanded','true');place();cancelAnimationFrame(trackingFrame);trackingFrame=requestAnimationFrame(follow)});
  details.addEventListener('toggle',()=>{if(!details.open){cancelAnimationFrame(trackingFrame);trackingFrame=0;if(menu.matches(':popover-open'))menu.hidePopover()}});
  document.addEventListener('pointerdown',e=>{if(details.open&&!trigger.contains(e.target)&&!menu.contains(e.target))close()},true);
  menu.addEventListener('click',()=>close());
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&details.open){close();trigger.focus()}});
  addEventListener('resize',()=>{if(details.open)place()});addEventListener('scroll',()=>{if(details.open)place()},true);
 });
})();

(() => {
 const cards=[...document.querySelectorAll('.campus-metrics article')];
 function align(){for(const card of cards){const number=card.querySelector('strong[id]');if(!number||!card.offsetHeight)continue;const box=card.getBoundingClientRect(),value=number.getBoundingClientRect(),scale=box.height/card.offsetHeight||1;card.style.setProperty('--metric-number-center',((value.top+value.height/2-box.top)/scale)+'px')}}
 if('ResizeObserver' in window){const observer=new ResizeObserver(align);cards.forEach(card=>{observer.observe(card);const number=card.querySelector('strong[id]');if(number)observer.observe(number)})}else addEventListener('resize',align);
 document.fonts?.ready.then(align);requestAnimationFrame(align);
})();
