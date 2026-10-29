/* Optional browser speech: no microphone permission or chatbot required. */
(() => {
  const button = document.getElementById('voiceToggle');
  const repeat = document.getElementById('voiceRepeat');
  const caption = document.getElementById('voiceCaption');
  const panel = document.getElementById('voicePanel');
  const available = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
  let enabled = false, text = 'Tap Enable voice guidance to hear scanning instructions.', key = '', last = 0, generation = 0;
  let lockedUntil = 0;
  const synth = available ? window.speechSynthesis : null;
  function paint() {
    button.textContent = enabled ? '🔊 Mute voice guidance' : '🔈 Enable voice guidance';
    button.setAttribute('aria-pressed', String(enabled));
    repeat.disabled = !enabled;
    document.getElementById('voiceState').textContent = enabled ? 'Audio guidance: On' : 'Audio guidance: Off';
  }
  function stop() { generation++; if(synth) synth.cancel(); panel.classList.remove('is-speaking'); }
  function speak() {
    if (!enabled || !available || document.hidden) return;
    stop(); const token = generation;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US'; utterance.rate = 0.95; utterance.pitch = 1;
    const voices = synth.getVoices();
    const voice = voices.find(v => v.localService && /^en[-_]/i.test(v.lang)) || voices.find(v => /^en[-_]/i.test(v.lang));
    if (voice) utterance.voice = voice;
    utterance.onstart = () => { if(token === generation) panel.classList.add('is-speaking'); };
    utterance.onend = () => { if(token === generation) panel.classList.remove('is-speaking'); };
    utterance.onerror = event => {
      if(token !== generation || event.error === 'canceled' || event.error === 'interrupted') return;
      panel.classList.remove('is-speaking'); enabled = false; paint();
      caption.textContent = text + ' Voice unavailable on this device; follow the written instructions.';
    };
    try { synth.speak(utterance); } catch (_) { enabled = false; paint(); }
  }
  function stage(index) {
    document.querySelectorAll('[data-scan-step]').forEach((el, i) => {
      el.classList.toggle('is-current', i === index);
      el.classList.toggle('is-done', i < index);
      if(i === index) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
    });
  }
  function prompt(message, newKey = message, priority = false) {
    const now = Date.now();
    if (!priority && now < lockedUntil) return;
    if (newKey === key) return; // A 500ms scan loop must not queue repeated speech.
    // Briefly changing position hints must settle before another utterance.
    if(!priority && now - last < 2300) return;
    key = newKey; text = message; last = now; caption.textContent = text;
    speak();
  }
  function result(data) {
    const name = String(data.user || '').trim().split(/\s+/)[0];
    const messages = {
      success: name ? 'Welcome, ' + name + '. Access granted.' : 'Welcome. Access granted.',
      denied: 'Access could not be verified. Please try again or contact security.',
      spoof: 'Access denied. Please use your real face and try again.',
      not_permitted: 'You are not permitted at this gate. Please use your assigned gate.',
      lockdown: 'Emergency lockdown is active. Please follow security instructions.',
      gate_locked: 'This gate is closed. Please use another gate.',
      busy: 'Please wait a moment before trying again.',
      error: 'The scan could not be completed. Please try again.'
    };
    if(messages[data.status]) {
      // A new completed scan may legitimately have the same result as the last.
      prompt(messages[data.status], 'result-' + data.status + '-' + Date.now(), true);
      lockedUntil = Date.now() + 4500; stage(2);
    }
  }
  window.AmanVoice = {
    prompt, result, stage,
    reset() { lockedUntil = 0; key = ''; last = 0; stage(0); prompt('Look at the camera and center your face.', 'ready', true); },
    stop() { stop(); lockedUntil = 0; key = ''; text = 'Camera closed. Tap Open Camera to begin.'; caption.textContent = text; stage(0); }
  };
  button.addEventListener('click', () => {
    enabled = !enabled; paint();
    if(enabled) { key = ''; last = 0; text = 'Voice guidance is on. Look at the camera to begin.'; caption.textContent = text; speak(); }
    else stop();
  });
  repeat.addEventListener('click', () => { if(enabled) speak(); });
  document.addEventListener('visibilitychange', () => { if(document.hidden) stop(); });
  window.addEventListener('pagehide', stop);
  if(!available) {
    button.disabled = true; repeat.disabled = true;
    document.getElementById('voiceState').textContent = 'Audio unavailable';
    caption.textContent = 'This browser does not support voice guidance. Follow the written instructions.';
  } else { synth.getVoices(); paint(); }
  stage(0);
})();
