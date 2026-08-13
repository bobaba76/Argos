// Auxiliary Models — Hermes desktop settings page (plugin).
//
// Exposes the config.yaml `auxiliary:` section in the desktop UI: per-task
// provider/model for every auxiliary task, plus the fallback policy
// (auxiliary.free_only, auxiliary.openrouter_model). Reads and writes via
// the gateway JSON-RPC doors `config.get` / `config.set` (the same RPC the
// app's own settings surfaces use — see apps/desktop/src/store/model-presets.ts).
//
// Plain ESM, loaded at runtime by the desktop app — no build step. It hot-
// reloads on save; if it fails to load, the app shows a toast naming the
// error. Install location: $HERMES_HOME/desktop-plugins/aux-models/plugin.js
// (a tracked copy lives in the customization bundle under desktop_plugins/).
import { jsx } from 'react/jsx-runtime'
import { useState, useEffect } from 'react'
import {
  host,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  Input,
  Switch,
  Button,
  Badge,
  Separator
} from '@hermes/plugin-sdk'

// The auxiliary tasks that carry explicit config entries. Any task NOT listed
// here (e.g. graph_entity_extraction) is auto-detected from the main provider
// and is not configurable — the page notes that below.
const TASKS = [
  { key: 'vision', label: 'Vision' },
  { key: 'web_extract', label: 'Web extract' },
  { key: 'compression', label: 'Context compression' },
  { key: 'skills_hub', label: 'Skills hub' },
  { key: 'approval', label: 'Approvals' },
  { key: 'mcp', label: 'MCP' },
  { key: 'title_generation', label: 'Title generation' },
  { key: 'curator', label: 'Curator' },
  { key: 'memory_extraction', label: 'Memory extraction' }
]

const GLOBAL_KEYS = ['auxiliary.free_only', 'auxiliary.openrouter_model']

function fieldKeys() {
  const keys = []
  for (const t of TASKS) {
    keys.push(`auxiliary.${t.key}.provider`)
    keys.push(`auxiliary.${t.key}.model`)
  }
  return keys.concat(GLOBAL_KEYS)
}

async function readConfig() {
  const out = {}
  for (const key of fieldKeys()) {
    try {
      const res = await host.request('config.get', { key })
      out[key] = res && res.value !== undefined ? String(res.value) : ''
    } catch {
      out[key] = ''
    }
  }
  return out
}

const rowStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '6px 0'
}
const labelStyle = {
  width: '190px',
  flex: '0 0 190px',
  fontSize: '12px',
  color: 'var(--ui-text-secondary)'
}
const inputStyle = { flex: '1 1 0', minWidth: 0 }

function TaskRow({ task, values, setValue }) {
  return jsx('div', {
    style: rowStyle,
    children: [
      jsx('span', { style: labelStyle, children: task.label }),
      jsx(Input, {
        value: values[`auxiliary.${task.key}.provider`] || '',
        placeholder: 'provider',
        onChange: e => setValue(`auxiliary.${task.key}.provider`, e.target.value)
      }),
      jsx(Input, {
        value: values[`auxiliary.${task.key}.model`] || '',
        placeholder: 'model',
        onChange: e => setValue(`auxiliary.${task.key}.model`, e.target.value)
      })
    ]
  })
}

function AuxModelsPage() {
  const [values, setValues] = useState({})
  const [loaded, setLoaded] = useState(false)
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState('')

  useEffect(() => {
    let cancelled = false
    readConfig().then(v => {
      if (!cancelled) {
        setValues(v)
        setLoaded(true)
      }
    })
    return () => { cancelled = true }
  }, [])

  const setValue = (key, value) => setValues(prev => ({ ...prev, [key]: value }))

  const save = async () => {
    setSaving(true)
    setStatus('')
    let ok = true
    try {
      for (const key of fieldKeys()) {
        const cur = await host.request('config.get', { key }).then(r => (r && r.value !== undefined ? String(r.value) : ''))
        const next = values[key] === undefined ? '' : String(values[key])
        if (cur !== next) {
          const res = await host.request('config.set', { key, value: next })
          if (res && res.ok === false) ok = false
        }
      }
    } catch (e) {
      ok = false
    }
    setSaving(false)
    setStatus(ok ? `Saved ${new Date().toLocaleTimeString()}` : 'Save failed — see console')
  }

  return jsx('div', {
    style: { padding: '16px 20px', maxWidth: '720px' },
    children: [
      jsx('h2', { style: { margin: '0 0 4px', fontSize: '16px' }, children: 'Auxiliary models' }),
      jsx('p', {
        style: { margin: '0 0 12px', fontSize: '12px', color: 'var(--ui-text-secondary)' },
        children: 'Providers/models for background tasks (extraction, titles, compression, approvals). Reads and writes the auxiliary section of config.yaml via the gateway.'
      }),
      jsx('div', { children: TASKS.map(t => jsx(TaskRow, { key: t.key, task: t, values, setValue })) }),
      jsx(Separator, { style: { margin: '10px 0' } }),
      jsx('div', {
        style: rowStyle,
        children: [
          jsx('span', { style: labelStyle, children: 'Free-only fallback' }),
          jsx(Switch, {
            checked: values['auxiliary.free_only'] === 'true',
            onCheckedChange: v => setValue('auxiliary.free_only', v ? 'true' : 'false')
          }),
          jsx('span', {
            style: { fontSize: '12px', color: 'var(--ui-text-quaternary)' },
            children: 'Restrict fallbacks to :free OpenRouter SKUs (no paid bleed)'
          })
        ]
      }),
      jsx('div', {
        style: rowStyle,
        children: [
          jsx('span', { style: labelStyle, children: 'OpenRouter fallback model' }),
          jsx(Input, {
            value: values['auxiliary.openrouter_model'] || '',
            placeholder: 'e.g. google/gemini-3.6-flash:free',
            style: inputStyle,
            onChange: e => setValue('auxiliary.openrouter_model', e.target.value)
          })
        ]
      }),
      jsx('div', {
        style: { ...rowStyle, gap: '12px' },
        children: [
          jsx(Button, { onClick: save, disabled: saving, children: saving ? 'Saving…' : 'Save changes' }),
          status ? jsx(Badge, { children: status }) : null
        ]
      }),
      jsx('p', {
        style: { marginTop: '12px', fontSize: '11px', color: 'var(--ui-text-quaternary)' },
        children: 'Tasks without an explicit entry here (e.g. graph_entity_extraction) are auto-detected from the main provider and can\u2019t be overridden per-task.'
      })
    ]
  })
}

export default {
  id: 'aux-models',
  name: 'Auxiliary Models',
  description: 'Settings page for auxiliary task providers/models and fallback policy (config.yaml auxiliary section).',
  register(ctx) {
    ctx.register({
      id: 'page',
      area: ROUTES_AREA,
      data: { path: '/aux-models' },
      render: () => jsx(AuxModelsPage, {})
    })
    ctx.register({
      id: 'nav',
      area: SIDEBAR_NAV_AREA,
      data: { path: '/aux-models', label: 'Aux Models', codicon: 'project' }
    })
  }
}
