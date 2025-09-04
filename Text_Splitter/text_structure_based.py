# from langchain.text_splitter import RecursiveCharacterTextSplitter

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=10,
#     chunk_overlap=0
# )

# text = """
# My name is Rishabh Singh Dhami
# I live in Kalagarh

# My age is 22
# My girlfriend name is Arushi Rana
# """
# result = splitter.split_text(text=text)
# print(len(result))


from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
export class Emitter {
  constructor() { this.events = new Map(); }
  on(evt, fn) {
    const list = this.events.get(evt) || [];
    list.push(fn); this.events.set(evt, list);
    return () => this.off(evt, fn);
  }
  off(evt, fn) {
    const list = this.events.get(evt) || [];
    const i = list.indexOf(fn);
    if (i > -1) list.splice(i, 1);
    if (!list.length) this.events.delete(evt);
  }
  emit(evt, ...args) {
    const list = this.events.get(evt) || [];
    for (const fn of [...list]) try { fn(...args); } catch (e) { console.error(e); }
  }
}

// LRU Cache
export class LRU {
  constructor(limit = 100) { this.limit = limit; this.map = new Map(); }
  get(key) {
    if (!this.map.has(key)) return undefined;
    const val = this.map.get(key);
    this.map.delete(key); this.map.set(key, val);
    return val;
  }
  set(key, val) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, val);
    if (this.map.size > this.limit) {
      const firstKey = this.map.keys().next().value;
      this.map.delete(firstKey);
    }
  }
  has(key) { return this.map.has(key); }
  delete(key) { return this.map.delete(key); }
  clear() { this.map.clear(); }
  get size() { return this.map.size; }
}

// Debounce
export const debounce = (fn, wait = 200, immediate = false) => {
  let t; return function(...args) {
    const callNow = immediate && !t;
    clearTimeout(t);
    t = setTimeout(() => { t = null; if (!immediate) fn.apply(this, args); }, wait);
    if (callNow) fn.apply(this, args);
  };
};

// Throttle (trailing edge)
export const throttle = (fn, wait = 200) => {
  let last = 0, t, queuedArgs;
  return function(...args) {
    const now = Date.now();
    const remaining = wait - (now - last);
    queuedArgs = args;
    if (remaining <= 0) {
      clearTimeout(t); t = null; last = now; fn.apply(this, queuedArgs);
    } else if (!t) {
      t = setTimeout(() => { last = Date.now(); t = null; fn.apply(this, queuedArgs); }, remaining);
    }
  };
};

// Exponential backoff with jitter
export async function retry(fn, { retries = 5, base = 200, factor = 2, jitter = true } = {}) {
  let attempt = 0, err;
  while (attempt <= retries) {
    try { return await fn(); }
    catch (e) {
      err = e;
      if (attempt === retries) break;
      const delay = Math.floor(base * (factor ** attempt) * (jitter ? (0.5 + Math.random()) : 1));
      await new Promise(r => setTimeout(r, delay));
      attempt++;
    }
  }
  throw err;
}

// Fetch wrapper with cache + retry
export function createClient({ ttlMs = 10_000, lruSize = 128 } = {}) {
  const emitter = new Emitter();
  const cache = new LRU(lruSize);
  const touch = (key, val) => cache.set(key, { v: val, t: Date.now() });

  async function get(url, opts = {}) {
    const key = JSON.stringify({ url, opts });
    const hit = cache.get(key);
    if (hit && Date.now() - hit.t < ttlMs) { emitter.emit("cacheHit", url); return hit.v; }
    emitter.emit("request", url);
    const res = await retry(() => fetch(url, opts).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }));
    touch(key, res);
    emitter.emit("response", url, res);
    return res;
  }

  return { get, on: (...a) => emitter.on(...a), off: (...a) => emitter.off(...a), cache };
}
"""


splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=80,
    chunk_overlap=0,
    language=Language.JS
)

chunks = splitter.split_text(text=text)
print(chunks[0])