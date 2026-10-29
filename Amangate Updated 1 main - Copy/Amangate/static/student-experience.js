(() => {
 const body=document.body, get=id=>document.getElementById(id);let approved=false;
 const gates={main:'Main Gate',male:'Male Gate',female:'Female Gate',vehicle:'Vehicle Gate'};
 function state(value){body.dataset.scanState=value}
 window.StudentExperience={
  camera(on){body.classList.toggle('camera-on',on);get('cameraBtn').textContent=on?'Close camera':'Open Camera'},
  ready(on){approved=false;state(on?'position':'ready');get('studentTitle').textContent='Welcome to campus';get('studentSubtitle').textContent=on?'Face the camera and hold still.':'Let’s get you through.';},
  message(text){if(approved)return;if(/blink/i.test(text)){get('studentTitle').textContent='Looking good!';get('studentSubtitle').textContent='Face the camera and hold still.';state('guided')}},
  tone(tone){if(tone==='danger'&&!approved)state('denied')},
  verifying(){state('verifying');get('studentTitle').textContent='Looking good!';get('studentSubtitle').textContent='Hold still while we verify your identity.'},
  result(data){if(data.status==='success'){approved=true;state('success');const name=String(data.user||'').trim().split(/\s+/)[0];get('studentTitle').textContent=name?'Welcome, '+name+'!':'Welcome!';get('studentSubtitle').textContent='Have a great day on campus.';get('successGate').textContent=gates[get('gateSelect').value]||'Main Gate'}else if(['denied','spoof','not_permitted','lockdown','gate_locked','error'].includes(data.status)){state('denied');get('studentTitle').textContent='Let’s check that.';get('studentSubtitle').textContent=data.status==='not_permitted'?'Please use your permitted gate.':'Follow the instructions below.'}}
 };
 get('continueSilent').onclick=()=>{if(get('voiceToggle').getAttribute('aria-pressed')==='true')get('voiceToggle').click();if(!body.classList.contains('camera-on'))get('cameraBtn').click()};
 get('voiceToggle').addEventListener('click',()=>{if(get('voiceToggle').getAttribute('aria-pressed')==='true'&&!body.classList.contains('camera-on'))get('cameraBtn').click()});
})();