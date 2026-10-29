/* Audio-only enhancement. Does not change the original terminal layout or scan rules. */
(() => {
 const button=document.getElementById('audioOnlyToggle'),caption=document.getElementById('audioOnlyCaption');
 if(!button||!caption)return;
 const supported='speechSynthesis' in window&&'SpeechSynthesisUtterance' in window;
 const synth=supported?window.speechSynthesis:null;
 // The on/off choice is remembered on this device (the terminal reloads when its gate changes)
 const preferenceKey='amangate.terminal.audio';
 function savedAudio(){try{return localStorage.getItem(preferenceKey)==='on'}catch{return false}}
 function saveAudio(){try{localStorage.setItem(preferenceKey,enabled?'on':'off')}catch{}}
 let enabled=supported&&savedAudio(),current='Audio guidance enabled. Open the camera to begin.',key='',last=0,hold=0,generation=0;
 function paint(){button.textContent=enabled?'Mute audio guidance':'Enable audio guidance';button.setAttribute('aria-pressed',String(enabled))}
 function cancel(){generation++;synth?.cancel()}
 function speak(){if(!enabled||!synth||document.hidden)return;cancel();const token=generation;const u=new SpeechSynthesisUtterance(current);u.lang='en-US';u.rate=.95;const voices=synth.getVoices();const voice=voices.find(v=>v.localService&&/^en[-_]/i.test(v.lang))||voices.find(v=>/^en[-_]/i.test(v.lang));if(voice)u.voice=voice;u.onerror=e=>{if(token!==generation||e.error==='canceled'||e.error==='interrupted')return;enabled=false;paint();caption.textContent='Voice unavailable. Follow the written scanning instructions.'};try{synth.speak(u)}catch{enabled=false;paint();caption.textContent='Voice unavailable. Follow the written scanning instructions.'}}
 function prompt(text,newKey=text,priority=false){const now=Date.now();if(!text||newKey===key||!priority&&(now<hold||now-last<2300))return;key=newKey;last=now;current=text;caption.textContent=text;speak()}
 window.AmanAudio={prompt,ready(){key='';last=0;hold=0;prompt('Look at the camera and center your face.','ready',true)},stop(){cancel();key='';hold=0;current='Open the camera to begin.';caption.textContent=current},result(data){const name=String(data.user||'').trim().split(/\s+/)[0];const texts={success:name?'Welcome, '+name+'. Access granted.':'Access granted.',denied:'Access could not be verified. Please try again or contact security.',spoof:'Access denied. Please use your real face.',not_permitted:'You are not permitted at this gate. Please use your assigned gate.',outside_hours:'You are outside your access hours. Please come back during your allowed time.',lockdown:'Emergency lockdown is active. Follow security instructions.',gate_locked:'This gate is closed. Please use another gate.',busy:'Please wait a moment before trying again.',error:'The scan could not be completed. Please try again.'};if(texts[data.status]){prompt(texts[data.status],'result-'+data.status+'-'+Date.now(),true);hold=Date.now()+4000}}};
 button.onclick=()=>{enabled=!enabled;saveAudio();paint();if(enabled){caption.textContent=current;speak()}else cancel()};
 document.addEventListener('visibilitychange',()=>{if(document.hidden)cancel()});window.addEventListener('pagehide',cancel);
 if(!supported){button.disabled=true;button.textContent='Audio unavailable';caption.textContent='Follow the written scanning instructions.'}else{synth.getVoices();paint();if(enabled)caption.textContent='Audio guidance is on. Open the camera to begin.'}
})();