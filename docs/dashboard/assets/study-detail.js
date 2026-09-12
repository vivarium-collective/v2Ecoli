// study-detail.js — wires the six-card Study Detail page to /api/study-* routes.
(function() {
  function api(method, path, body) {
    return fetch(path, {
      method: method,
      headers: body ? {'Content-Type': 'application/json'} : {},
      body: body ? JSON.stringify(body) : null,
    }).then(function(r) {
      return r.json().then(function(d) { return {status: r.status, body: d}; });
    });
  }

  // --- Tab navigation ---
  function _setStudyTab(kind) {
    document.querySelectorAll('.study-tab').forEach(function(b) {
      b.classList.toggle('active', b.dataset.kind === kind);
    });
    document.querySelectorAll('.study-tab-panel').forEach(function(p) {
      p.classList.toggle('active', p.dataset.kind === kind);
    });
    if (kind === 'tests') {
      loadTestsTab(window._study);
    }
    if (kind === 'conclusions') {
      _loadConclusionsTab(window._study);
    }
    if (kind === 'visualizations') {
      _loadCharts('viz-charts-panel');
    }
  }
  window._setStudyTab = _setStudyTab;

  // ── Charts panel: inline SVGs from /api/study-charts ─────────────────────
  // Lives in the Visualizations tab only. Memoized per panel id.
  // Merges two sources returned by the server:
  //   live   — generated from runs.db at request time
  //   static — pre-rendered SVGs under studies/<name>/charts/
  var _chartsLoadedFor = {};
  function _renderChartCard(c) {
    var title = c.title
      ? '<div class="chart-title">' + c.title + '</div>'
      : '';
    // SVG records carry inline markup in c.svg; PNG/GIF records carry a
    // self-contained data-URI in c.img (rendered as <img>).
    var media = c.img
      ? '<img class="chart-img" src="' + c.img + '" alt="' + (c.key || 'chart') + '" loading="lazy">'
      : (c.svg || '');
    return '<div class="chart-card">' + title + media +
           '<div class="chart-caption">' + (c.caption || '') + '</div></div>';
  }
  function _loadCharts(panelId) {
    if (_chartsLoadedFor[panelId]) return;
    var panel = document.getElementById(panelId);
    if (!panel) return;
    _chartsLoadedFor[panelId] = true;
    // In snapshot mode the /api/study-charts/ endpoint is not exported.
    // Show a neutral placeholder instead of attempting the fetch.
    var _cfg = window.__DASH_CONFIG__ || {};
    if (_cfg.mode === 'snapshot') {
      panel.innerHTML = '<p class="muted" style="margin:0">Results are served by viva-api — coming in a later update.</p>';
      return;
    }
    panel.innerHTML = '<p class="muted" style="margin:0">Loading charts…</p>';
    fetch('/api/study-charts/' + encodeURIComponent(studyName()))
      .then(function(r) { return r.json(); })
      .then(function(d) {
        if (!d || !d.charts || !d.charts.length) {
          panel.innerHTML = (d && d.db_exists === false)
            ? '<p class="muted" style="margin:0">No <code>runs.db</code> and no static charts under <code>studies/' + studyName() + '/charts/</code>.</p>'
            : '<p class="muted" style="margin:0">No chart data available for this study.</p>';
          return;
        }
        var live = d.charts.filter(function(c) { return (c.source || 'live') === 'live'; });
        var stat = d.charts.filter(function(c) { return c.source === 'static'; });
        var html = '';
        if (live.length) {
          html += live.map(_renderChartCard).join('');
        }
        if (stat.length) {
          if (live.length) {
            html += '<h3 class="section-title" style="margin-top:24px">Pre-rendered charts <span class="muted" style="font-weight:400;font-size:0.85em">(checked-in under <code>studies/' + studyName() + '/charts/</code>)</span></h3>';
          }
          html += stat.map(_renderChartCard).join('');
        }
        panel.innerHTML = html;
      })
      .catch(function(e) {
        panel.innerHTML = '<p class="muted" style="color:#dc2626">Chart load failed: ' + (e && e.message || e) + '</p>';
      });
  }

  // ── Seed a new study from a follow_up_studies[] entry ────────────────────
  function _seedFollowupStudy(parentStudyName, followupIdx) {
    if (!confirm('Seed a new study from this follow-up?\n\nA new study.yaml will be created under studies/<new-name>/ pre-populated with the follow-up context.')) {
      return;
    }
    api('POST', '/api/study-seed-followup', {parent: parentStudyName, followup_idx: followupIdx})
      .then(function(res) {
        if (res.status !== 200 || res.body.error) {
          alert('Seed failed: ' + (res.body.error || res.status));
          return;
        }
        alert('Created: ' + res.body.new_study_name + '\nOpening it now.');
        window.location.href = '/studies/' + encodeURIComponent(res.body.new_study_name);
      });
  }
  window._seedFollowupStudy = _seedFollowupStudy;

  // --- Inline-edit (overview fields: objective, conclusion, question, hypothesis, status) ---
  function _saveOverviewField(field, value) {
    if (field === 'objective') {
      return api('POST', '/api/study-set-objective', {study: studyName(), text: value});
    }
    if (field === 'conclusion') {
      return api('POST', '/api/study-set-conclusion', {study: studyName(), text: value});
    }
    if (field === 'question' || field === 'hypothesis' || field === 'status') {
      var body = {investigation: studyName(), fields: {}};
      body.fields[field] = value;
      return api('POST', '/api/investigation-set-overview', body);
    }
    return Promise.resolve();
  }

  // --- Conclusions tab: split/join helpers + load/save ---
  function _splitConclusion(md) {
    var sections = { Claims: '', Evidence: '', Limitations: '', 'Next steps': '' };
    if (!md) return sections;
    var parts = md.split(/(?:^|\n)##\s+/);
    if (parts.length === 1) {
      sections.Claims = parts[0].trim();
      return sections;
    }
    var preamble = parts.shift();
    if (preamble && preamble.trim()) sections.Claims = preamble.trim();
    parts.forEach(function(chunk) {
      var nl = chunk.indexOf('\n');
      var header = (nl === -1 ? chunk : chunk.slice(0, nl)).trim();
      var body = (nl === -1 ? '' : chunk.slice(nl + 1)).trim();
      if (header in sections) {
        if (sections[header]) sections[header] += '\n\n' + body;
        else sections[header] = body;
      }
    });
    return sections;
  }

  function _joinConclusion(sections) {
    var labels = ['Claims', 'Evidence', 'Limitations', 'Next steps'];
    var parts = labels.map(function(label) {
      var body = (sections[label] || '').trim();
      return '## ' + label + (body ? '\n\n' + body : '');
    });
    return parts.join('\n\n') + '\n';
  }

  function _loadConclusionsTab(study) {
    var s = _splitConclusion((study && study.conclusion) || '');
    var ids = { Claims: 'conclusion-claims', Evidence: 'conclusion-evidence',
                Limitations: 'conclusion-limitations', 'Next steps': 'conclusion-next-steps' };
    Object.keys(ids).forEach(function(label) {
      var el = document.getElementById(ids[label]);
      if (el) el.value = s[label] || '';
    });
  }

  function _saveConclusion() {
    var sections = {
      Claims:       (document.getElementById('conclusion-claims') || {}).value || '',
      Evidence:     (document.getElementById('conclusion-evidence') || {}).value || '',
      Limitations:  (document.getElementById('conclusion-limitations') || {}).value || '',
      'Next steps': (document.getElementById('conclusion-next-steps') || {}).value || '',
    };
    return fetch('/api/study-set-conclusion', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({study: studyName(), text: _joinConclusion(sections)}),
    });
  }

  function makeEditable(el) {
    if (!el) return;
    var placeholder = el.dataset.placeholder || '';
    var field = el.dataset.field || el.id.replace(/-text$/, '');
    el.addEventListener('click', function() {
      if (el.querySelector('textarea')) return;
      var current = el.textContent.trim();
      var t = document.createElement('textarea');
      t.value = (current === placeholder) ? '' : current;
      t.rows = 4;
      t.style.width = '100%';
      el.innerHTML = '';
      el.appendChild(t);
      t.focus();
      t.addEventListener('blur', function() {
        _saveOverviewField(field, t.value).then(function() {
          el.textContent = t.value || placeholder;
        });
      });
    });
  }

  document.querySelectorAll('[data-editable="true"]').forEach(function(el) {
    makeEditable(el);
  });

  // --- v4 narrative-spine forms: report / study_card / biological_summary /
  // conclusion_verdicts. Every [data-narrative-path] input saves to the
  // generic /api/study-narrative-set on blur (text/textarea) or change
  // (select). The path is a dotted route into the v4 narrative-spine
  // sub-tree; the backend resolves it, creates parents as needed, and
  // atomically writes study.yaml.
  function _saveNarrative(el) {
    var path = el.dataset.narrativePath;
    if (!path) return;
    var value = el.value;
    el.classList.remove('narrative-saved', 'narrative-error');
    return api('POST', '/api/study-narrative-set', {
      study: studyName(),
      path: path,
      value: value,
    }).then(function(res) {
      // api() returns {status, body}. 200 + body.ok === success.
      if (res && res.status === 200 && res.body && res.body.ok) {
        el.classList.add('narrative-saved');
        setTimeout(function() { el.classList.remove('narrative-saved'); }, 700);
      } else {
        el.classList.add('narrative-error');
        var detail = (res && res.body && res.body.error) || (res && res.status) || 'unknown';
        el.title = 'Save failed: ' + detail;
      }
    }).catch(function(e) {
      el.classList.add('narrative-error');
      el.title = 'Network error: ' + (e && e.message || e);
    });
  }
  document.querySelectorAll('[data-narrative-path]').forEach(function(el) {
    var tag = (el.tagName || '').toLowerCase();
    // Selects save on change (immediate, no need to wait for blur). Text
    // inputs + textareas save on blur so the user can type without round-
    // tripping per keystroke.
    var evt = (tag === 'select') ? 'change' : 'blur';
    el.addEventListener(evt, function() { _saveNarrative(el); });
  });

  var statusSel = document.getElementById('status-select');
  if (statusSel) {
    statusSel.addEventListener('change', function() {
      _saveOverviewField('status', statusSel.value);
    });
  }

  ['conclusion-claims', 'conclusion-evidence', 'conclusion-limitations', 'conclusion-next-steps'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener('blur', _saveConclusion);
  });

  // --- Helpers: attach a click handler to every button matching a CSS class ---
  function bindAll(selector, handler) {
    document.querySelectorAll(selector).forEach(function(btn) {
      btn.addEventListener('click', function(ev) { handler(btn, ev); });
    });
  }

  function studyName() { return window._studyName; }

  // Fetch the param schema for a composite and render an input form.
  // currentOverrides: {} or existing overrides (for edit flow).
  // Returns a Promise<{collect, ok}>: collect() reads back the current input
  // values and returns an overrides dict; ok=false if fetch failed (containerEl
  // shows the error message in that case).
  function renderParamForm(containerEl, specId, currentOverrides) {
    var overridesJson = encodeURIComponent(JSON.stringify(currentOverrides || {}));
    return fetch('/api/composite-resolve?id=' +
                 encodeURIComponent(specId) + '&overrides=' + overridesJson)
      .then(function(r) { return r.json().then(function(b) { return {status: r.status, body: b}; }); })
      .then(function(r) {
        if (r.status !== 200) {
          containerEl.innerHTML = '<p class="error">Could not resolve composite: ' +
            (r.body && r.body.error || r.status) + '</p>';
          return {collect: function() { return {}; }, ok: false};
        }
        var params = r.body.parameters || {};
        containerEl.innerHTML = '';
        var inputs = {};
        Object.keys(params).forEach(function(k) {
          var def = params[k] || {};
          var type = def.type || 'string';
          var current = (currentOverrides && k in currentOverrides) ? currentOverrides[k] : def.default;
          var row = document.createElement('div');
          row.className = 'param-row';
          var label = document.createElement('label');
          label.className = 'param-label';
          var nameSpan = document.createElement('span');
          nameSpan.innerHTML = '<code>' + k + '</code> <span class="muted">(' + type + ')</span>';
          var input = document.createElement('input');
          input.className = 'param-input';
          input.dataset.paramKey = k;
          input.dataset.paramType = type;
          if (type === 'integer' || type === 'number' || type === 'float') {
            input.type = 'number';
            input.step = (type === 'integer') ? '1' : 'any';
          } else if (type === 'boolean') {
            input.type = 'checkbox';
            if (current === true) input.checked = true;
          } else {
            input.type = 'text';
          }
          if (input.type !== 'checkbox' && current !== undefined && current !== null) {
            input.value = current;
          }
          label.appendChild(nameSpan);
          label.appendChild(input);
          row.appendChild(label);
          if (def.description) {
            var desc = document.createElement('div');
            desc.className = 'param-desc muted';
            desc.textContent = def.description;
            row.appendChild(desc);
          }
          containerEl.appendChild(row);
          inputs[k] = input;
        });
        var collect = function() {
          var out = {};
          Object.keys(inputs).forEach(function(k) {
            var el = inputs[k];
            var t = el.dataset.paramType;
            if (t === 'boolean') out[k] = !!el.checked;
            else if (t === 'integer') out[k] = el.value === '' ? null : parseInt(el.value, 10);
            else if (t === 'number' || t === 'float') out[k] = el.value === '' ? null : parseFloat(el.value);
            else out[k] = el.value;
          });
          // Remove null/empty entries (don't send them as overrides).
          Object.keys(out).forEach(function(k) {
            if (out[k] === null || out[k] === '' || out[k] === undefined) delete out[k];
          });
          return out;
        };
        return {collect: collect, ok: true};
      });
  }
  // Not exposed on window; consumed internally by the Variants tab.

  // --- Header actions ---
  bindAll('.btn-rename', function() {
    var n = prompt('New name (lowercase + dashes):', studyName());
    if (!n) return;
    // study-rename handler (_post_study_rename_for_test) uses body key "study"
    api('POST', '/api/study-rename', {study: studyName(), new_name: n})
      .then(function(res) {
        if (res.status === 200) window.location = '/studies/' + n;
        else alert(res.body.error || 'Rename failed');
      });
  });

  bindAll('.btn-export', function() {
    window.location = '/api/study-export?study=' + encodeURIComponent(studyName());
  });

  // btn-delete has class "btn-delete danger" — selector ".btn-delete" still matches.
  // Handler _post_investigation_delete uses body key "name".
  bindAll('.btn-delete', function(btn) {
    // Guard: only the header delete button has data-study; variant/run deletes
    // use different class names so this handler won't fire for those.
    if (!btn.dataset.study) return;
    if (!confirm('Delete this study and all its runs?')) return;
    api('POST', '/api/study-delete', {name: studyName(), study: studyName()})
      .then(function() { window.location = '/studies'; });
  });

  // --- Baseline ---
  bindAll('.btn-run-baseline', function(btn) {
    var entryName = btn.dataset.baselineName;
    api('POST', '/api/study-run-baseline', {
      study: studyName(), composite: entryName
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Run failed: ' + (r.body && r.body.error || r.status));
    });
  });

  bindAll('.btn-baseline-remove', function(btn) {
    var entryName = btn.dataset.baselineName;
    if (!confirm('Remove baseline composite "' + entryName + '"?')) return;
    api('POST', '/api/study-baseline-remove', {
      study: studyName(), name: entryName
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else if (r.status === 409 && r.body.dependents) {
        alert('Cannot remove: variants depend on this composite (' +
              r.body.dependents.join(', ') + '). Delete those variants first.');
      } else {
        alert('Remove failed: ' + (r.body && r.body.error || r.status));
      }
    });
  });

  function _submitBaselineAdd(ev) {
    ev.preventDefault();
    var form = ev.target;
    var params = {};
    var raw = form.params.value.trim();
    if (raw) {
      try { params = JSON.parse(raw); }
      catch (e) { alert('Params must be valid JSON.'); return false; }
    }
    api('POST', '/api/study-baseline-add', {
      study: studyName(),
      name: form.name.value.trim(),
      composite: form.composite.value.trim(),
      params: params
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Add failed: ' + (r.body && r.body.error || r.status));
    });
    return false;
  }
  window._submitBaselineAdd = _submitBaselineAdd;

  // ===== Variants tab handlers =====

  // Variant add — base-composite dropdown changes trigger param-form render.
  var _currentVariantAddCollect = null;
  function _onBaseCompositeChange(selectEl) {
    var opt = selectEl.options[selectEl.selectedIndex];
    var specId = opt.dataset.compositeId || '';
    var container = document.getElementById('variant-new-params');
    if (!specId) {
      container.innerHTML = '<p class="muted">Pick a base composite to see its parameters.</p>';
      _currentVariantAddCollect = null;
      return;
    }
    container.innerHTML = '<p class="muted">Loading parameters…</p>';
    renderParamForm(container, specId, {}).then(function(result) {
      _currentVariantAddCollect = result.collect;
    });
  }
  window._onBaseCompositeChange = _onBaseCompositeChange;

  function _submitVariantAdd(ev) {
    ev.preventDefault();
    var form = ev.target;
    var name = form.name.value.trim();
    var baseComposite = form.base_composite.value;
    if (!name || !baseComposite) { alert('Name and base composite are required.'); return false; }
    var overrides = _currentVariantAddCollect ? _currentVariantAddCollect() : {};
    api('POST', '/api/study-variant-add', {
      study: studyName(), name: name, base_composite: baseComposite,
      parameter_overrides: overrides
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Add variant failed: ' + (r.body && r.body.error || r.status));
    });
    return false;
  }
  window._submitVariantAdd = _submitVariantAdd;

  // Variant edit — populate dialog, render param form with current overrides, then save.
  var _currentVariantEditCollect = null;
  bindAll('.btn-variant-edit', function(btn) {
    var variantName = btn.dataset.variantName;
    var variant = (window._study.variants || []).filter(function(v) { return v.name === variantName; })[0];
    if (!variant) { alert('Variant not found in local spec.'); return; }
    var baseEntry = (window._study.baseline || []).filter(function(b) { return b.name === variant.base_composite; })[0];
    if (!baseEntry) { alert('Variant references a base composite that no longer exists.'); return; }
    document.getElementById('variant-edit-name').textContent = variantName;
    var container = document.getElementById('variant-edit-params');
    container.innerHTML = '<p class="muted">Loading parameters…</p>';
    renderParamForm(container, baseEntry.composite, variant.parameter_overrides || {}).then(function(result) {
      _currentVariantEditCollect = result.collect;
      document.getElementById('variant-edit-dialog').dataset.variantName = variantName;
      document.getElementById('variant-edit-dialog').showModal();
    });
  });

  function _submitVariantEdit(ev) {
    ev.preventDefault();
    var dialog = document.getElementById('variant-edit-dialog');
    var variantName = dialog.dataset.variantName;
    var overrides = _currentVariantEditCollect ? _currentVariantEditCollect() : {};
    api('POST', '/api/study-variant-set-params', {
      study: studyName(), variant: variantName, parameter_overrides: overrides
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Save failed: ' + (r.body && r.body.error || r.status));
    });
    return false;
  }
  window._submitVariantEdit = _submitVariantEdit;

  bindAll('.btn-variant-delete', function(btn) {
    var variantName = btn.dataset.variantName;
    if (!confirm('Delete variant "' + variantName + '"?')) return;
    api('POST', '/api/study-variant-delete', {
      study: studyName(), variant: variantName
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Delete failed: ' + (r.body && r.body.error || r.status));
    });
  });

  bindAll('.btn-variant-run', function(btn) {
    var variantName = btn.dataset.variantName;
    api('POST', '/api/study-run-variant', {
      study: studyName(), variant: variantName
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Run failed: ' + (r.body && r.body.error || r.status));
    });
  });

  // ===== Interventions tab handlers =====

  function _submitInterventionAdd(ev) {
    ev.preventDefault();
    var form = ev.target;
    api('POST', '/api/study-intervention-add', {
      study: studyName(),
      name: form.name.value.trim(),
      description: form.description.value
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Add failed: ' + (r.body && r.body.error || r.status));
    });
    return false;
  }
  window._submitInterventionAdd = _submitInterventionAdd;

  bindAll('.btn-intervention-delete', function(btn) {
    var name = btn.dataset.interventionName;
    if (!confirm('Delete intervention "' + name + '"?')) return;
    api('POST', '/api/study-intervention-delete', {
      study: studyName(), name: name
    }).then(function(r) {
      if (r.status === 200) location.reload();
      else alert('Delete failed: ' + (r.body && r.body.error || r.status));
    });
  });

  // Inline-edit intervention descriptions. Uses a click-to-textarea pattern
  // parallel to makeEditable but POSTs to the intervention-update endpoint.
  document.querySelectorAll('[data-editable-intervention]').forEach(function(el) {
    el.addEventListener('click', function() {
      var name = el.dataset.editableIntervention;
      var current = el.textContent;
      var t = document.createElement('textarea');
      t.value = current;
      t.style.width = '100%';
      t.rows = 3;
      el.replaceWith(t);
      t.focus();
      t.addEventListener('blur', function() {
        api('POST', '/api/study-intervention-update', {
          study: studyName(), name: name, description: t.value
        }).then(function(r) {
          if (r.status === 200) location.reload();
          else { alert('Update failed: ' + (r.body && r.body.error || r.status)); }
        });
      });
    });
  });

  // --- Runs ---
  bindAll('.btn-view-run', function(btn) {
    var runId = btn.dataset.runId;
    window.open('/composite-explorer?run_id=' + encodeURIComponent(runId), '_blank');
  });

  // ptools-launch → _get_ptools_launch
  bindAll('.btn-launch-ptools', function(btn) {
    var runId = btn.dataset.runId;
    var study = studyName();
    var url = '/api/ptools-launch/' + encodeURIComponent(study) + '?run=' + encodeURIComponent(runId);
    fetch(url).then(function(r) {
      return r.json().then(function(d) { return {status: r.status, body: d}; });
    }).then(function(res) {
      var b = res.body;
      if (res.status === 200 && b.url) {
        window.open(b.url, '_blank');
      } else if (b && b.error === 'ptools_server_url not configured') {
        alert('PTools not configured.\nSet ui.ptools_server_url in workspace.yaml.');
      } else if (b && b.available && b.available.length === 0) {
        alert('No ptools TSV results found for this run.\nRun the ptools analyses first.');
      } else {
        alert('PTools launch failed: ' + (b && b.error || res.status));
      }
    });
  });

  // study-run-delete → _post_investigation_run_delete
  bindAll('.btn-delete-run', function(btn) {
    var runId = btn.dataset.runId;
    if (!confirm('Delete this run?')) return;
    api('POST', '/api/study-run-delete', {
      study: studyName(), run_id: runId,
    }).then(function() { location.reload(); });
  });

  // study-runs-clear → _post_investigation_runs_clear
  bindAll('.btn-clear-runs', function() {
    if (!confirm('Clear ALL runs in this study?')) return;
    api('POST', '/api/study-runs-clear', {
      study: studyName(),
    }).then(function() { location.reload(); });
  });

  // study-comparison-add → _post_investigation_comparison_add
  bindAll('.btn-compare-selected', function() {
    var ids = [];
    document.querySelectorAll('.run-compare-checkbox:checked').forEach(function(c) {
      ids.push(c.value);
    });
    if (ids.length < 2) return alert('Select at least two runs.');
    api('POST', '/api/study-comparison-add', {
      study: studyName(), run_ids: ids,
    }).then(function(res) {
      if (res.status === 200) location.reload();
      else alert(res.body.error || 'Compare failed');
    });
  });

  // --- Viz ---
  // NOTE: .btn-view-run intentionally left as-is (broken URL is a follow-up task).
  bindAll('.btn-add-viz', function() {
    // The add-viz modal lives on the main dashboard page. Take the user there.
    location.href = '/#composite-explore?study=' + encodeURIComponent(studyName());
  });

  // --- Conclusion ---
  // study-set-conclusion → _post_investigation_set_conclusions, key "investigation"
  // but also aliased to set-conclusion which uses "investigation" key.
  bindAll('.btn-mark-complete', function() {
    api('POST', '/api/study-set-conclusion', {
      investigation: studyName(), study: studyName(), mark_complete: true,
    }).then(function() { location.reload(); });
  });

  // ----- Tests tab -----

  function loadTestsTab(spec) {
    var cfg = (spec && spec.tests) || {};
    var autoEl = document.getElementById('tests-auto-discover');
    var dsEl = document.getElementById('tests-data-source');
    if (autoEl) autoEl.textContent = String(cfg.auto_discover !== undefined ? cfg.auto_discover : true);
    if (dsEl) dsEl.textContent = cfg.data_source || 'latest_run';
    var summary = document.getElementById('tests-summary');
    if (!summary) return;

    // Prefer aggregated outcomes from runs[].outcomes (the v3-shape result
    // recording), falling back to legacy tests.last_results.
    var passed = 0, failed = 0, skipped = 0, runRefs = 0;
    (spec && spec.runs || []).forEach(function(r) {
      if (!r.outcomes) return;
      runRefs++;
      Object.keys(r.outcomes).forEach(function(tname) {
        var res = (r.outcomes[tname] || {}).result;
        if (res === 'PASS') passed++;
        else if (res === 'FAIL') failed++;
        else if (res === 'SKIP') skipped++;
      });
    });

    if (passed + failed + skipped > 0) {
      var lastRun = (spec.runs || [])[spec.runs.length - 1] || {};
      summary.innerHTML =
        '<span class="ok">' + passed + ' passed</span>' +
        ' / <span class="fail">' + failed + ' failed</span>' +
        ' / <span class="skip">' + skipped + ' skipped</span>' +
        ' <span class="muted">(' + runRefs + ' run' + (runRefs === 1 ? '' : 's') + ' recorded; latest: ' +
        (lastRun.started_at || '?') + ')</span>';
    } else if (cfg.last_results) {
      var lr = cfg.last_results;
      summary.innerHTML =
        '<span class="ok">' + (lr.passed || 0) + ' passed</span>' +
        ' / <span class="fail">' + (lr.failed || 0) + ' failed</span>' +
        ' / <span class="skip">' + (lr.skipped || 0) + ' skipped</span>' +
        ' <span class="muted">(' + ((lr.duration_s || 0).toFixed(2)) + 's' +
        (lr.timestamp ? ', ' + lr.timestamp : '') + ')</span>';
    } else {
      summary.textContent = '— no test results yet — click "Run tests" to execute them or check the runs[] section in study.yaml';
    }

    // --- Code-computed outcomes summary ---
    // Tally spec.runs[].computed_outcomes (written by the post-run evaluator).
    // Skip the _status sentinel key and any non-object entries.
    var cPassed = 0, cFailed = 0, cAgent = 0;
    var cAgree = 0, cDivergent = 0, cNoAuthored = 0;
    var anyComputed = false;
    (spec && spec.runs || []).forEach(function(r) {
      var co = r.computed_outcomes;
      if (!co || typeof co !== 'object' || Array.isArray(co)) return;
      Object.keys(co).forEach(function(tname) {
        if (tname === '_status') return;
        var entry = co[tname];
        if (!entry || typeof entry !== 'object') return;
        anyComputed = true;
        var evaluatedBy = entry.evaluated_by || '';
        if (evaluatedBy === 'code') {
          if (entry.result === 'PASS') cPassed++;
          else if (entry.result === 'FAIL') cFailed++;
          else cAgent++;
        } else {
          cAgent++;
        }
        var reconcile = entry.reconcile || '';
        if (reconcile === 'agree') cAgree++;
        else if (reconcile === 'divergent') cDivergent++;
        else if (reconcile === 'no_authored') cNoAuthored++;
      });
    });

    if (anyComputed) {
      var compEl = document.getElementById('tests-computed-summary');
      if (!compEl) {
        compEl = document.createElement('div');
        compEl.id = 'tests-computed-summary';
        compEl.className = 'tests-summary muted';
        summary.insertAdjacentElement('afterend', compEl);
      }
      var cHtml =
        '<span class="muted">Code-computed: </span>' +
        '<span class="ok">' + cPassed + ' passed</span>' +
        ' / <span class="fail">' + cFailed + ' failed</span>' +
        ' / <span class="muted">' + cAgent + ' agent</span>';
      if (cDivergent > 0) {
        cHtml += '  <span class="fail" style="font-weight:600">' +
          '⚠ ' + escapeHtmlForTests(String(cDivergent)) +
          ' divergent from authored</span>';
      }
      var muted = [];
      if (cAgree > 0) muted.push(escapeHtmlForTests(String(cAgree)) + ' agree');
      if (cNoAuthored > 0) muted.push(escapeHtmlForTests(String(cNoAuthored)) + ' no_authored');
      if (muted.length) {
        cHtml += ' <span class="muted">(' + muted.join(', ') + ')</span>';
      }
      compEl.innerHTML = cHtml;
    }
  }

  function escapeHtmlForTests(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function(c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }

  function renderTestResults(body) {
    var list = document.getElementById('tests-list');
    if (!list) return;
    list.innerHTML = '';
    if (body.note === 'no tests directory') {
      list.innerHTML = '<li class="placeholder">No tests/ directory found in this study.</li>';
      return;
    }
    var icons = {passed: '✅', failed: '❌', skipped: '⏭'};
    (body.tests || []).forEach(function(t) {
      var li = document.createElement('li');
      li.className = 'test-row test-' + t.outcome;
      var icon = icons[t.outcome] || '•';
      var tb = t.traceback
        ? '<details><summary>detail</summary><pre>' + escapeHtmlForTests(t.traceback) + '</pre></details>'
        : '';
      var dur = t.duration
        ? '<span class="test-duration">' + (t.duration).toFixed(3) + 's</span>' : '';
      li.innerHTML =
        '<span class="test-icon">' + icon + '</span>' +
        '<code class="test-nodeid">' + escapeHtmlForTests(t.nodeid) + '</code>' +
        dur + tb;
      list.appendChild(li);
    });
    var s = body.summary || {};
    var summary = document.getElementById('tests-summary');
    if (summary) {
      summary.innerHTML =
        '<span class="ok">' + (s.passed || 0) + ' passed</span>' +
        ' / <span class="fail">' + (s.failed || 0) + ' failed</span>' +
        ' / <span class="skip">' + (s.skipped || 0) + ' skipped</span>' +
        ' <span class="muted">(' + ((s.duration_s || 0).toFixed(2)) + 's)</span>' +
        (body.note ? ' <span class="muted" style="font-style:italic">— ' + escapeHtmlForTests(body.note) + '</span>' : '');
    }
  }

  function runStudyTests() {
    var btn = document.getElementById('run-tests-btn');
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = 'Running…';
    fetch('/api/study-tests-run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({study: studyName()}),
    }).then(function(resp) {
      return resp.json().then(function(d) { return {status: resp.status, body: d}; });
    }).then(function(r) {
      if (r.status !== 200) {
        alert('Test run failed: ' + (r.body && r.body.error || r.status));
        return;
      }
      renderTestResults(r.body);
    }).catch(function(err) {
      alert('Test run error: ' + err);
    }).then(function() {
      btn.disabled = false;
      btn.textContent = 'Run tests';
    });
  }

  var runBtn = document.getElementById('run-tests-btn');
  if (runBtn) {
    runBtn.addEventListener('click', runStudyTests);
  }

  // ── Stage-3c: Tracked Feedback panel ─────────────────────────────────────
  // Renders open/addressed/dismissed items from window._study.feedback_tracked
  // into #feedback-tracked-panel (Overview tab).  Idempotent — skips if already
  // populated.  Escapes all user-supplied text.  Renders nothing when empty.
  // Pure render, no AI.
  function _esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function(c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }

  function _renderFeedbackTrackedPanel() {
    var container = document.getElementById('feedback-tracked-panel');
    if (!container) return;               // anchor missing — template version mismatch
    if (container.dataset.rendered) return; // idempotent
    container.dataset.rendered = '1';

    var spec = window._study || {};
    var ft = spec.feedback_tracked;
    if (!ft || !ft.items || ft.items.length === 0) return;  // nothing to show

    var summary = ft.summary || {};
    var openCt  = summary.open      || 0;
    var addrCt  = summary.addressed || 0;
    var disCt   = summary.dismissed || 0;
    var total   = summary.total     || ft.items.length;

    // Status badge colours
    var badgeCss = {
      open:      'background:#fef3c7;color:#92400e;',
      addressed: 'background:#d1fae5;color:#065f46;',
      dismissed: 'background:#f1f5f9;color:#64748b;text-decoration:line-through;',
    };

    var itemsHtml = '';
    (ft.items || []).forEach(function(item) {
      var status   = item.status || 'open';
      var badgeStyle = badgeCss[status] || badgeCss.open;
      var badgeHtml  =
        '<span style="' + badgeStyle +
        'padding:1px 8px;border-radius:9999px;font-size:0.78em;' +
        'font-family:ui-monospace,monospace;margin-right:6px">' +
        _esc(status) + '</span>';

      var metaHtml =
        '<span class="muted" style="font-size:0.82em">' +
        _esc(item.author || '') + ' · ' + _esc((item.ts || '').replace('T', ' ').replace('Z', ' UTC')) +
        ' · <code style="font-size:0.9em">' + _esc(item.section || '') + '</code>' +
        '</span>';

      var textHtml = '<p style="margin:4px 0;font-size:0.92em">' + _esc(item.text || '') + '</p>';

      var responseHtml = '';
      if (status === 'addressed' && item.response) {
        responseHtml =
          '<div style="margin:6px 0 0 0;padding:8px 12px;background:#f0fdf4;' +
          'border-left:3px solid #10b981;border-radius:4px;font-size:0.88em">' +
          '<strong style="font-size:0.85em;color:#065f46">Response' +
          (item.responded_by ? ' (' + _esc(item.responded_by) + ')' : '') +
          (item.responded_at ? ' — ' + _esc(item.responded_at) : '') +
          ':</strong>' +
          '<pre style="white-space:pre-wrap;margin:4px 0 0 0;font-family:inherit;' +
          'font-size:0.92em;color:#374151">' + _esc(item.response) + '</pre>' +
          '</div>';
      }

      itemsHtml +=
        '<div style="padding:10px 14px;border-bottom:1px solid #f1f5f9">' +
        '<div style="display:flex;align-items:flex-start;gap:6px;flex-wrap:wrap;margin-bottom:4px">' +
        badgeHtml + metaHtml +
        '</div>' +
        textHtml +
        responseHtml +
        '</div>';
    });

    var summaryHtml =
      '<span style="font-size:0.9em">' +
      '<span style="color:#92400e">' + openCt + ' open</span>' +
      ' / <span style="color:#065f46">' + addrCt + ' addressed</span>' +
      ' / <span style="color:#64748b">' + disCt + ' dismissed</span>' +
      ' <span class="muted">(' + total + ' total)</span>' +
      '</span>';

    container.innerHTML =
      '<div class="overview-section" style="margin-top:18px">' +
      '<h2 class="overview-label">Expert Feedback</h2>' +
      '<div style="margin-bottom:10px">' + summaryHtml + '</div>' +
      '<div style="border:1px solid #e2e8f0;border-radius:6px;overflow:hidden">' +
      itemsHtml +
      '</div>' +
      '</div>';
  }

  // ── DataSource bootstrap (client-fetch seam, sub-project #1) ─────────────
  // Populate window._study via a fetch when the Jinja embed is absent.
  // The renderers (loadTestsTab, _loadConclusionsTab, _renderFeedbackTrackedPanel,
  // etc.) are unchanged — they still read window._study.  Only acquisition changes.

  function _showStudyLoadError(e) {
    var el = document.getElementById('study-root') || document.body;
    el.innerHTML =
      '<div style="padding:2rem;color:#dc2626">' +
      'Could not load study data: ' + String(e && e.message || e) +
      '</div>';
  }

  async function _bootstrapStudy() {
    if (!window._study && window.DataSource && window._studyName) {
      try {
        window._study = await window.DataSource.loadStudy(window._studyName);
      } catch (e) {
        _showStudyLoadError(e);
        return false;
      }
    }
    return !!window._study;
  }

  function _runStudyInit() {
    // All renderers that need window._study to be populated.
    _renderFeedbackTrackedPanel();
  }

  // Entry point: fetch the spec if needed, then run init.
  (async function () {
    if (await _bootstrapStudy()) { _runStudyInit(); }
  })();

})();
