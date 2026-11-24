(() => {
  'use strict';

  const state = {
    currentPage: 'arena',
    stage: 'name',
    roundId: null,
    playerName: '',
    pickedSlot: null,
    referenceUrl: '',
    isSubmitting: false,
    isPicking: false,
    lastRoundId: null,
    countdownInterval: null,
    failedTimeout: null,
    isFailed: false,
    secondsLeft: 60,
    isKioskLocked: false,
    isEnteringKiosk: false,
    kioskSecretBuffer: '',
  };

  const PROMPT_TIME_LIMIT_SECONDS = 60;
  const FAILED_DISPLAY_MS = 5000;
  const RESULT_FAILED_DISPLAY_MS = 4000;
  const LOW_SIMILARITY_FAIL_THRESHOLD = 30;
  const KIOSK_SECRET_PHRASE = 'utmsanjunganbangsa';

  let toastTimer = null;

  function $(id) {
    return document.getElementById(id);
  }

  function showToast(message) {
    const toast = $('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove('visible');
    }, 3200);
  }

  async function apiJson(url, options = {}) {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
    return data;
  }

  const appStartTime = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  const MIN_LOADING_TIME = 1500;

  function hideLoader() {
    const loader = $('app-loading');
    if (!loader) return;
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const elapsed = now - appStartTime;
    const delay = Math.max(0, MIN_LOADING_TIME - elapsed);
    setTimeout(() => {
      loader.classList.add('hidden');
      setTimeout(() => {
        if (loader.parentNode) loader.parentNode.removeChild(loader);
      }, 400);
    }, delay);
  }

  function closeSidebar() {
    $('side-panel').classList.remove('open');
    $('panel-overlay').classList.remove('visible');
    $('hamburger').classList.remove('active');
  }

  function navigateTo(page) {
    const valid = ['arena', 'leaderboard'];
    if (!valid.includes(page)) page = 'arena';
    state.currentPage = page;
    window.location.hash = page;

    document.querySelectorAll('.nav-link').forEach((el) => {
      el.classList.toggle('active', el.dataset.page === page);
    });
    document.querySelectorAll('.page').forEach((el) => {
      el.classList.toggle('active', el.id === `page-${page}`);
    });

    const mainEl = document.querySelector('.main-content');
    if (mainEl) mainEl.scrollTop = 0;

    closeSidebar();

    if (page !== 'arena') {
      stopPromptCountdown();
      clearChallengeFailed();
    }

    if (page === 'leaderboard') {
      loadLeaderboard(false);
    } else if (page === 'arena' && state.stage === 'name') {
      loadLeaderboard(true);
    }
  }

  function bindNavigation() {
    document.querySelectorAll('.nav-link').forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        navigateTo(link.dataset.page);
      });
    });

    $('hamburger').addEventListener('click', () => {
      const panel = $('side-panel');
      const isOpen = panel.classList.toggle('open');
      $('panel-overlay').classList.toggle('visible', isOpen);
      $('hamburger').classList.toggle('active', isOpen);
    });

    $('panel-overlay').addEventListener('click', closeSidebar);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeSidebar();
      }
    });

    window.addEventListener('hashchange', () => {
      const page = window.location.hash.replace('#', '') || 'arena';
      navigateTo(page);
    });
  }

  const STAGE_ORDER = ['name', 'pick', 'prompt', 'result'];

  function isKioskUrl() {
    return new URLSearchParams(window.location.search).get('kiosk') === '1';
  }

  function isKioskActive() {
    return state.isKioskLocked || isKioskUrl();
  }

  function setStage(stage, refreshLeaderboard = true) {
    state.stage = stage;
    state.kioskSecretBuffer = '';
    const targetIdx = STAGE_ORDER.indexOf(stage);

    document.querySelectorAll('.stage').forEach((el) => {
      el.classList.toggle('active', el.id === `stage-${stage}`);
    });

    document.querySelectorAll('#arena-stepper .stepper__item').forEach((item) => {
      const idx = STAGE_ORDER.indexOf(item.dataset.step);
      item.classList.toggle('active', idx === targetIdx);
      item.classList.toggle('done', idx < targetIdx);
    });

    const mainEl = document.querySelector('.main-content');
    if (mainEl) mainEl.scrollTop = 0;

    const hero = $('arena-hero');
    if (hero) {
      hero.classList.toggle('is-collapsed', stage !== 'name');
    }

    const toolbarRow = $('stage-toolbar-row');
    const pickActions = $('pick-toolbar-actions');
    if (state.roundId && stage !== 'name') {
      toolbarRow.style.display = 'flex';
      if (pickActions) {
        pickActions.style.display = stage === 'pick' ? 'flex' : 'none';
      }
      $('active-player-name').textContent = state.playerName.toUpperCase();
    } else {
      toolbarRow.style.display = 'none';
      if (pickActions) {
        pickActions.style.display = 'none';
      }
    }

    if (stage === 'name' && refreshLeaderboard) {
      loadLeaderboard(true);
    }
    if (stage !== 'prompt') {
      stopPromptCountdown();
    }
  }

  function startNewRound() {
    clearChallengeFailed();
    state.roundId = null;
    state.pickedSlot = null;
    state.referenceUrl = '';
    $('player-name-input').value = '';
    if (state.currentPage !== 'arena') {
      setStage('name', false);
      navigateTo('arena');
    } else {
      setStage('name');
    }
    if (!isKioskActive()) {
      $('player-name-input').focus();
    } else if (document.activeElement && document.activeElement.blur) {
      document.activeElement.blur();
    }
  }

  function bindStageName() {
    $('player-name-input').addEventListener('input', () => {
      if (!isKioskActive()) return;
      const val = $('player-name-input').value.trim().toLowerCase();
      if (val === KIOSK_SECRET_PHRASE) {
        $('player-name-input').value = '';
        unlockKioskMode(KIOSK_SECRET_PHRASE);
      }
    });

    $('form-start-round').addEventListener('submit', async (e) => {
      e.preventDefault();
      const input = $('player-name-input');
      const name = input.value.trim();
      if (!name) {
        showToast('Please enter your name before proceeding.');
        input.focus();
        return;
      }

      const btn = $('btn-start-round');
      btn.disabled = true;
      btn.textContent = 'SHUFFLING 10 IMAGES...';

      try {
        const data = await apiJson('/api/round/start', {
          method: 'POST',
          body: JSON.stringify({ player_name: name }),
        });
        state.roundId = data.round_id;
        state.playerName = data.player_name;
        state.pickedSlot = null;
        state.referenceUrl = '';
        renderBlindGrid(data.slots || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], true);
        setStage('pick');
      } catch (err) {
        showToast(err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'ENTER CHALLENGE';
      }
    });
  }

  function renderBlindGrid(slots, animate = false) {
    const grid = $('blind-grid');
    grid.innerHTML = '';

    slots.forEach((slotNum, idx) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'mystery-card' + (animate ? ' is-shuffling' : '');
      card.style.animationDelay = animate ? `${idx * 35}ms` : '0ms';
      card.dataset.slot = String(slotNum);
      card.setAttribute('aria-label', `Pick hidden image card ${slotNum}`);

      const center = document.createElement('div');
      center.className = 'mystery-card__center';
      const num = document.createElement('span');
      num.className = 'mystery-card__number';
      num.textContent = String(slotNum);
      center.appendChild(num);

      card.appendChild(center);
      card.addEventListener('click', () => handleBlindPick(slotNum, card));
      grid.appendChild(card);
    });
  }

  async function handleBlindPick(slotNum, cardEl) {
    if (state.isPicking || !state.roundId) return;
    state.isPicking = true;
    cardEl.classList.add('is-selected');

    try {
      const data = await apiJson(`/api/round/${encodeURIComponent(state.roundId)}/pick`, {
        method: 'POST',
        body: JSON.stringify({ slot_index: slotNum }),
      });
      state.pickedSlot = data.picked_slot;
      state.referenceUrl = data.reference_image_url;

      $('prompt-slot-badge').textContent = `SLOT #${String(data.picked_slot).padStart(2, '0')} - ${data.challenge_code}`;
      $('reference-image').src = data.reference_image_url;
      $('prompt-input').value = '';
      $('prompt-char-count').textContent = '0 / 800';
      $('generating-box').style.display = 'none';
      $('form-submit-prompt').style.display = 'flex';
      $('countdown-widget').style.display = 'flex';

      setTimeout(() => {
        setStage('prompt');
        startPromptCountdown();
        $('prompt-input').focus();
      }, 180);
    } catch (err) {
      cardEl.classList.remove('is-selected');
      showToast(err.message);
    } finally {
      state.isPicking = false;
    }
  }

  function bindStagePick() {
    $('btn-reshuffle').addEventListener('click', async () => {
      if (!state.roundId || state.isPicking) return;
      try {
        const data = await apiJson(`/api/round/${encodeURIComponent(state.roundId)}/reshuffle`, {
          method: 'POST',
        });
        renderBlindGrid(data.slots, true);
        showToast('10 images randomly reshuffled.');
      } catch (err) {
        showToast(err.message);
      }
    });

    $('btn-change-player').addEventListener('click', () => {
      if (state.isPicking) return;
      state.roundId = null;
      setStage('name');
      $('player-name-input').focus();
    });
  }

  function stopPromptCountdown() {
    if (state.countdownInterval) {
      clearInterval(state.countdownInterval);
      state.countdownInterval = null;
    }
  }

  function clearChallengeFailed() {
    if (state.failedTimeout) {
      clearTimeout(state.failedTimeout);
      state.failedTimeout = null;
    }
    state.isFailed = false;
    const overlay = $('challenge-failed-overlay');
    if (overlay) {
      overlay.classList.remove('visible');
      overlay.setAttribute('aria-hidden', 'true');
    }
  }

  function triggerChallengeFailed(options = {}) {
    const {
      durationMs = FAILED_DISPLAY_MS,
      redirectToNewRound = true,
    } = options;
    if (state.isFailed) return;
    stopPromptCountdown();
    state.isFailed = true;
    const textarea = $('prompt-input');
    if (textarea) textarea.blur();
    const overlay = $('challenge-failed-overlay');
    if (overlay) {
      overlay.classList.remove('visible');
      void overlay.offsetWidth;
      overlay.setAttribute('aria-hidden', 'false');
      overlay.classList.add('visible');
    }
    state.failedTimeout = setTimeout(() => {
      clearChallengeFailed();
      if (redirectToNewRound) {
        startNewRound();
      }
    }, durationMs);
  }

  function renderCountdownValue(val) {
    const el = $('countdown-seconds');
    if (!el) return;
    el.textContent = String(val);
    el.classList.toggle('is-urgent', val <= 10);
  }

  function startPromptCountdown() {
    stopPromptCountdown();
    state.secondsLeft = PROMPT_TIME_LIMIT_SECONDS;
    renderCountdownValue(state.secondsLeft);

    state.countdownInterval = setInterval(() => {
      if (state.stage !== 'prompt' || state.isSubmitting) {
        stopPromptCountdown();
        return;
      }
      state.secondsLeft = Math.max(0, state.secondsLeft - 1);
      renderCountdownValue(state.secondsLeft);
      if (state.secondsLeft <= 0) {
        stopPromptCountdown();
        const prompt = $('prompt-input').value.trim();
        if (!prompt) {
          triggerChallengeFailed();
        } else {
          submitPromptRound(true);
        }
      }
    }, 1000);
  }

  async function submitPromptRound(autoSubmit = false) {
    if (state.isSubmitting || state.isFailed || !state.roundId) return;

    const textarea = $('prompt-input');
    const prompt = textarea.value.trim();
    if (autoSubmit && prompt.length === 0) {
      triggerChallengeFailed();
      return;
    }
    if (!autoSubmit && prompt.length < 3) {
      showToast('Please write a descriptive prompt (at least 3 characters).');
      textarea.focus();
      return;
    }

    stopPromptCountdown();
    state.isSubmitting = true;
    const btn = $('btn-submit-prompt');
    btn.disabled = true;

    $('form-submit-prompt').style.display = 'none';
    $('countdown-widget').style.display = 'none';
    $('generating-box').style.display = 'block';

    const statusMessages = [
      autoSubmit ? "Time's up — generating image from your prompt..." : 'Generating image from your prompt...',
      'Comparing ResNet50 semantic & neural style features...',
      'Computing CIE-LAB palette & spatial layout score...',
    ];
    let msgIdx = 0;
    $('generating-status-text').textContent = statusMessages[0];
    const msgTimer = setInterval(() => {
      msgIdx = Math.min(msgIdx + 1, statusMessages.length - 1);
      $('generating-status-text').textContent = statusMessages[msgIdx];
    }, 2200);

    try {
      const result = await apiJson(`/api/round/${encodeURIComponent(state.roundId)}/submit`, {
        method: 'POST',
        body: JSON.stringify({ prompt, auto_submit: autoSubmit }),
      });
      state.lastRoundId = result.round_id;
      await renderResultStage(result);
      setStage('result');
      refreshLeaderboardBadge();
      if (Number(result.similarity && result.similarity.overall) <= LOW_SIMILARITY_FAIL_THRESHOLD) {
        triggerChallengeFailed({ durationMs: RESULT_FAILED_DISPLAY_MS, redirectToNewRound: false });
      }
    } catch (err) {
      $('generating-box').style.display = 'none';
      $('form-submit-prompt').style.display = 'flex';
      if (!autoSubmit) {
        $('countdown-widget').style.display = 'flex';
      }
      showToast(err.message);
    } finally {
      clearInterval(msgTimer);
      state.isSubmitting = false;
      btn.disabled = false;
    }
  }

  function bindStagePrompt() {
    const textarea = $('prompt-input');
    const counter = $('prompt-char-count');

    textarea.addEventListener('input', () => {
      counter.textContent = `${textarea.value.length} / 800`;
    });

    textarea.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        submitPromptRound(false);
      }
    });

    $('btn-zoom-ref').addEventListener('click', () => {
      if (!state.referenceUrl) return;
      $('lightbox-img').src = state.referenceUrl;
      $('lightbox-overlay').classList.add('visible');
    });

    $('lightbox-close').addEventListener('click', () => {
      $('lightbox-overlay').classList.remove('visible');
    });

    $('lightbox-overlay').addEventListener('click', (e) => {
      if (e.target === $('lightbox-overlay')) {
        $('lightbox-overlay').classList.remove('visible');
      }
    });

    $('form-submit-prompt').addEventListener('submit', async (e) => {
      e.preventDefault();
      await submitPromptRound(false);
    });
  }

  async function renderResultStage(result) {
    const sim = result.similarity;
    $('result-overall-score').textContent = `${sim.overall.toFixed(1)}%`;
    $('result-overall-bar').style.width = `${Math.max(0, Math.min(100, sim.overall))}%`;

    $('result-player-name').textContent = result.player_name;
    const rankLabel = `#${result.rank} OF ${result.total_entries}` + (result.rank <= 3 ? ' (TOP 3)' : '');
    $('result-rank-badge').textContent = rankLabel;
    $('result-engine-source').textContent = result.generator_source;
    $('result-slot-code').textContent = `SLOT #${String(result.picked_slot).padStart(2, '0')} (${result.challenge_code})`;

    $('result-ref-img').src = result.reference_image_url;
    $('result-gen-img').src = `${result.generated_image_url}?t=${Date.now()}`;
    $('result-prompt-text').textContent = result.prompt_text;

    $('score-semantic').textContent = `${sim.semantic.toFixed(1)}%`;
    $('bar-semantic').style.width = `${sim.semantic}%`;
    $('score-style').textContent = `${sim.style.toFixed(1)}%`;
    $('bar-style').style.width = `${sim.style}%`;
    $('score-color').textContent = `${sim.color.toFixed(1)}%`;
    $('bar-color').style.width = `${sim.color}%`;
    $('score-composition').textContent = `${sim.composition.toFixed(1)}%`;
    $('bar-composition').style.width = `${sim.composition}%`;

    try {
      const lbData = await apiJson('/api/leaderboard?limit=10');
      renderLeaderboardRows($('result-leaderboard-container'), lbData.entries.slice(0, 10), result.round_id, true);
    } catch (_) {
    }
  }

  function bindStageResult() {
    $('btn-play-next-round').addEventListener('click', startNewRound);

    $('btn-view-full-leaderboard').addEventListener('click', () => {
      navigateTo('leaderboard');
    });
  }

  function buildLeaderboardRow(entry, currentRoundId, compact = false) {
    const row = document.createElement('div');
    let cls = compact ? 'lb-row lb-row--compact' : 'lb-row';
    if (entry.rank === 1) cls += ' lb-row--top-1';
    else if (entry.rank === 2) cls += ' lb-row--top-2';
    else if (entry.rank === 3) cls += ' lb-row--top-3';
    if (currentRoundId && entry.round_id === currentRoundId) cls += ' lb-row--current';
    row.className = cls;

    const rankEl = document.createElement('div');
    rankEl.className = 'lb-rank';
    rankEl.textContent = String(entry.rank).padStart(2, '0');
    if (entry.rank <= 3) {
      const tag = document.createElement('span');
      tag.className = 'lb-rank-tag';
      tag.textContent = entry.rank === 1 ? '1ST' : entry.rank === 2 ? '2ND' : '3RD';
      rankEl.appendChild(tag);
    }

    const playerEl = document.createElement('div');
    playerEl.className = 'lb-player';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'lb-player__name';
    nameSpan.textContent = entry.player_name;
    playerEl.appendChild(nameSpan);

    if (currentRoundId && entry.round_id === currentRoundId) {
      const youTag = document.createElement('span');
      youTag.className = 'lb-you-tag';
      youTag.textContent = 'YOU';
      playerEl.appendChild(youTag);
    }

    const scoreEl = document.createElement('div');
    scoreEl.className = 'lb-score';
    scoreEl.textContent = `${Number(entry.similarity_score).toFixed(1)}%`;

    if (compact) {
      row.append(rankEl, playerEl, scoreEl);
      return row;
    }

    const slotEl = document.createElement('div');
    slotEl.className = 'lb-slot';
    slotEl.textContent = entry.challenge_code;

    const barWrap = document.createElement('div');
    barWrap.className = 'lb-bar-wrap';
    const barFill = document.createElement('div');
    barFill.className = 'lb-bar-fill';
    barFill.style.width = `${Math.max(0, Math.min(100, entry.similarity_score))}%`;
    barWrap.appendChild(barFill);

    row.append(rankEl, playerEl, slotEl, barWrap, scoreEl);
    return row;
  }

  function renderLeaderboardRows(container, entries, currentRoundId = null, compact = false) {
    container.innerHTML = '';
    if (!entries || entries.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      const title = document.createElement('div');
      title.className = 'empty-state__title';
      title.textContent = 'No Submissions Yet';
      const desc = document.createElement('div');
      desc.textContent = 'Complete a blind prompt round to claim the #1 spot on the leaderboard.';
      empty.append(title, desc);
      container.appendChild(empty);
      return;
    }

    const list = document.createElement('div');
    list.className = 'leaderboard-list';
    entries.forEach((entry) => {
      list.appendChild(buildLeaderboardRow(entry, currentRoundId, compact));
    });
    container.appendChild(list);
  }

  function renderTopThreePodium(entries) {
    const podium = $('podium-container');
    podium.innerHTML = '';
    if (!entries || entries.length === 0) {
      podium.style.display = 'none';
      return;
    }

    podium.style.display = 'grid';
    const topThree = entries.slice(0, 3);
    const labels = ['1ST PLACE', '2ND PLACE', '3RD PLACE'];

    topThree.forEach((entry, idx) => {
      const card = document.createElement('div');
      card.className = `podium-card podium-card--rank-${idx + 1}`;

      const top = document.createElement('div');
      top.className = 'podium-card__top';
      const rankNum = document.createElement('span');
      rankNum.textContent = `RANK 0${idx + 1}`;
      const badge = document.createElement('span');
      badge.className = 'podium-badge';
      badge.textContent = labels[idx];
      top.append(rankNum, badge);

      const name = document.createElement('div');
      name.className = 'podium-card__name';
      name.textContent = entry.player_name;

      const score = document.createElement('div');
      score.className = 'podium-card__score';
      score.textContent = `${Number(entry.similarity_score).toFixed(1)}%`;

      const meta = document.createElement('div');
      meta.className = 'podium-card__meta';
      meta.textContent = `${entry.challenge_code} // ${entry.generator_source}`;

      const imageCode = document.createElement('div');
      imageCode.className = 'podium-card__code';
      imageCode.textContent = String(entry.round_id || '').toLowerCase();

      card.append(top, name, score, meta, imageCode);
      podium.appendChild(card);
    });
  }

  async function loadLeaderboard(topOnly = false) {
    try {
      const data = await apiJson(topOnly ? '/api/leaderboard?limit=10' : '/api/leaderboard');
      $('nav-leaderboard-count').textContent = String(data.total);
      const homeContainer = $('home-leaderboard-container');
      if (homeContainer) {
        renderLeaderboardRows(homeContainer, data.entries.slice(0, 10), state.lastRoundId, true);
      }
      if (!topOnly) {
        renderTopThreePodium(data.entries);
        renderLeaderboardRows($('leaderboard-table-container'), data.entries, state.lastRoundId, false);
      }
    } catch (err) {
      showToast(err.message);
    }
  }

  async function refreshLeaderboardBadge() {
    try {
      const data = await apiJson('/api/leaderboard?limit=1');
      $('nav-leaderboard-count').textContent = String(data.total);
    } catch (_) { }
  }

  function bindLeaderboardActions() {
    $('btn-refresh-leaderboard').addEventListener('click', () => {
      loadLeaderboard();
      showToast('Leaderboard refreshed.');
    });

    $('btn-go-play').addEventListener('click', startNewRound);
  }

  async function loadEngineStatus() {
    try {
      const status = await apiJson('/api/config/engine');
      $('engine-status-text').textContent = status.active_mode;
    } catch (_) {
      $('engine-status-text').textContent = 'Offline Ready';
    }
  }

  function bindHostModal() {
    const overlay = $('host-modal-overlay');
    $('btn-open-host-modal').addEventListener('click', () => {
      closeSidebar();
      overlay.classList.add('visible');
    });
    $('host-modal-close').addEventListener('click', () => {
      overlay.classList.remove('visible');
    });
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.classList.remove('visible');
    });

    $('form-host-keys').addEventListener('submit', async (e) => {
      e.preventDefault();
      try {
        const res = await apiJson('/api/config/engine', {
          method: 'POST',
          body: JSON.stringify({
            hf_token: $('key-hf').value || null,
            cloudflare_account_id: $('key-cf-account').value || null,
            cloudflare_api_token: $('key-cf-token').value || null,
            gemini_api_key: $('key-gemini').value || null,
            openai_api_key: $('key-openai').value || null,
          }),
        });
        $('engine-status-text').textContent = res.active_mode;
        overlay.classList.remove('visible');
        showToast('Engine settings saved.');
      } catch (err) {
        showToast(err.message);
      }
    });

    $('btn-reset-leaderboard').addEventListener('click', async () => {
      if (!window.confirm('Clear all scores from the leaderboard? This cannot be undone.')) {
        return;
      }
      try {
        await apiJson('/api/leaderboard/reset', { method: 'POST' });
        state.lastRoundId = null;
        await loadLeaderboard();
        overlay.classList.remove('visible');
        showToast('Leaderboard cleared.');
      } catch (err) {
        showToast(err.message);
      }
    });
  }

  async function enterKioskMode() {
    if (state.isEnteringKiosk || isKioskActive()) return;
    state.isEnteringKiosk = true;
    state.kioskSecretBuffer = '';
    try {
      const res = await apiJson('/api/kiosk/enter', {
        method: 'POST',
        body: JSON.stringify({ origin: window.location.origin }),
      });
      if (!res.launched) {
        throw new Error('Could not launch Chrome kiosk mode.');
      }
    } catch (_) {
      try {
        if (document.documentElement.requestFullscreen) {
          await document.documentElement.requestFullscreen();
        }
        if (navigator.keyboard && navigator.keyboard.lock) {
          await navigator.keyboard.lock(['Escape', 'Tab']);
        }
      } catch (_) { }
      state.isKioskLocked = true;
      state.kioskSecretBuffer = '';
      if (document.activeElement && document.activeElement.blur) {
        document.activeElement.blur();
      }
    } finally {
      state.isEnteringKiosk = false;
    }
  }

  async function unlockKioskMode(phrase) {
    if (!isKioskActive()) return;
    state.kioskSecretBuffer = '';
    try {
      await apiJson('/api/kiosk/exit', {
        method: 'POST',
        body: JSON.stringify({ phrase }),
      });
    } catch (_) {
      return;
    }
    state.isKioskLocked = false;
    try {
      if (navigator.keyboard && navigator.keyboard.unlock) {
        navigator.keyboard.unlock();
      }
    } catch (_) { }
    try {
      if (document.fullscreenElement && document.exitFullscreen) {
        await document.exitFullscreen();
      }
    } catch (_) { }
    if (isKioskUrl()) {
      const url = new URL(window.location.href);
      url.searchParams.delete('kiosk');
      window.history.replaceState({}, '', url.pathname + url.search + url.hash);
    }
    showToast('Kiosk mode unlocked.');
  }

  function isBlockedKioskShortcut(e) {
    const key = (e.key || '').toLowerCase();
    const code = e.code || '';
    const fixedBlocked = [
      'f11', 'f12', 'f5', 'f3', 'f7', 'escape',
      'browserback', 'browserforward', 'browserrefresh',
      'browserstop', 'browsersearch', 'browserfavorites', 'browserhome',
    ];
    if (fixedBlocked.includes(key)) {
      return true;
    }
    if (e.altKey && (key === 'arrowleft' || key === 'arrowright' || key === 'home')) {
      return true;
    }
    if (e.ctrlKey && !e.altKey) {
      if (key === 'tab' || code === 'Tab' || key === 'pageup' || key === 'pagedown' || key === 'f4' || key === 'f5') {
        return true;
      }
      const blockedCtrlKeys = ['t', 'w', 'n', 'l', 'k', 'r', 'o', 'p', 's', 'u', 'h', 'j', 'd', 'g', 'f'];
      if (blockedCtrlKeys.includes(key)) {
        return true;
      }
      if (e.shiftKey && ['i', 'j', 'c', 'k', 't', 'n', 'w', 'delete'].includes(key)) {
        return true;
      }
    }
    return false;
  }

  function bindKioskGuard() {
    document.addEventListener('contextmenu', (e) => {
      if (isKioskActive()) {
        e.preventDefault();
      }
    }, true);

    document.addEventListener('auxclick', (e) => {
      if (isKioskActive() && e.button === 1) {
        e.preventDefault();
      }
    }, true);

    window.addEventListener('keydown', (e) => {
      if (!isKioskActive()) {
        if (e.key === 'F11' && !e.ctrlKey && !e.altKey && !e.metaKey) {
          e.preventDefault();
          e.stopPropagation();
          enterKioskMode();
        }
        return;
      }

      if (isBlockedKioskShortcut(e)) {
        e.preventDefault();
        e.stopPropagation();
        if (e.key === 'Escape' && $('lightbox-overlay').classList.contains('visible')) {
          $('lightbox-overlay').classList.remove('visible');
        }
        return;
      }

      const onIdentityPage = state.currentPage === 'arena' && state.stage === 'name';
      if (!onIdentityPage) {
        state.kioskSecretBuffer = '';
        return;
      }

      const hostOverlay = $('host-modal-overlay');
      if (hostOverlay && hostOverlay.classList.contains('visible')) {
        state.kioskSecretBuffer = '';
        return;
      }

      const activeEl = document.activeElement;
      const inEditable = activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA');
      if (inEditable) {
        state.kioskSecretBuffer = '';
        return;
      }

      if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key && e.key.length === 1) {
        const ch = e.key.toLowerCase();
        if (/^[a-z0-9]$/.test(ch)) {
          state.kioskSecretBuffer = (state.kioskSecretBuffer + ch).slice(-KIOSK_SECRET_PHRASE.length);
          if (state.kioskSecretBuffer === KIOSK_SECRET_PHRASE) {
            e.preventDefault();
            e.stopPropagation();
            unlockKioskMode(KIOSK_SECRET_PHRASE);
          }
        } else {
          state.kioskSecretBuffer = '';
        }
      } else if (e.key === 'Backspace' || e.key === 'Enter') {
        state.kioskSecretBuffer = '';
      }
    }, true);
  }

  function init() {
    if (isKioskUrl()) {
      state.isKioskLocked = true;
    }
    bindKioskGuard();
    bindNavigation();
    bindStageName();
    bindStagePick();
    bindStagePrompt();
    bindStageResult();
    bindLeaderboardActions();
    bindHostModal();

    const initialHash = window.location.hash.replace('#', '') || 'arena';
    navigateTo(initialHash);
    setStage('name');
    loadEngineStatus();
    refreshLeaderboardBadge();
    hideLoader();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
