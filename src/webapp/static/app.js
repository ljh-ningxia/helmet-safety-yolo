const $ = (s) => document.querySelector(s);

async function refreshState() {
  try {
    const s = await (await fetch('/api/state')).json();
    const m = s.metrics || {};
    $('#kpis').innerHTML = [
      ['已处理帧', m.frames ?? 0], ['本帧检出框', m.detections ?? 0],
      ['本帧确认违规', m.confirmed ?? 0], ['平均时延(ms)', m.latency_ms ?? 0],
      ['估算 FPS', m.fps ?? 0], ['告警事件', m.events ?? 0],
    ].map(([k, v]) => `<div class="kpi"><b>${v}</b><span>${k}</span></div>`).join('');
    $('#tick').textContent = '第 ' + (m.frames ?? 0) + ' 帧';
    $('#changed').textContent = '当前改动：' + (s.changed_modules || []).join(' ｜ ');
    const e = s.environment || {};
    $('#env').textContent = `环境指纹 ${e.hash || '-'} ｜ Python ${e.python || '-'} ｜ OpenCV ${e.opencv || '-'} ｜ 检测后端 ${s.detector?.backend || '-'}`;
    if (s.evidence_level === 'MEASURED') {
      $('#notice').style.display = 'none';
    }
    const evs = s.events || [];
    $('#events').innerHTML = evs.length ? evs.map(ev => {
      const cls = (ev.level || 'L1').replace(' ', '');
      return `<div class="ev"><span class="lv ${cls}">${ev.level}</span>
        <div class="txt">${ev.message}<br><span class="tm">${ev.time} ｜ 轨迹 ${ev.track_id} ｜ 置信度 ${ev.conf}</span></div></div>`;
    }).join('') : '<div class="empty">暂无告警事件</div>';
  } catch (err) { /* 服务刚启动时忽略 */ }
}

async function refreshAB() {
  try {
    const d = await (await fetch('/api/ab')).json();
    let html = '<table><thead><tr><th>实验</th><th>唯一主要变动</th><th>A 基线</th><th>B 改进</th><th>状态 / 边界</th></tr></thead><tbody>';
    for (const e of d.experiments) {
      if (e.status === '待测') {
        html += `<tr><td><b>${e.experiment}</b></td><td>—</td><td>—</td><td>—</td>
          <td><span class="pill wait">待测 · 运行 run.py ab 后填充</span></td></tr>`;
        continue;
      }
      const A = e.A || {}, B = e.B || {};
      html += `<tr><td><b>${e.title}</b><br><span class="pill ${e.evidence_level === 'MEASURED' ? 'ok' : 'wait'}">${e.evidence_level}</span></td>
        <td>${e.single_change}</td>
        <td>检出 ${A.total_detections}<br>FPS ${A.fps}<br>事件 ${A.alert_events}</td>
        <td>检出 ${B.total_detections}<br>FPS ${B.fps}<br>事件 ${B.alert_events}</td>
        <td>${e.conclusion_boundary}</td></tr>`;
    }
    $('#ab').innerHTML = html + '</tbody></table>';
  } catch (err) { /* 忽略 */ }
}

async function pushSwitches() {
  const body = {
    tiling: $('#sw-tiling').checked,
    temporal: $('#sw-temporal').checked,
    dedup: $('#sw-dedup').checked,
  };
  await fetch('/api/switch', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  refreshState();
}

['sw-tiling', 'sw-temporal', 'sw-dedup'].forEach(id =>
  document.getElementById(id).addEventListener('change', pushSwitches));

refreshState(); refreshAB();
setInterval(refreshState, 1200);
setInterval(refreshAB, 15000);
