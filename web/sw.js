// Service Worker for 佛学智能问答 PWA
const CACHE_NAME = 'rag-chat-v1';
const SHELL_URLS = [
  '/chat',
  '/web/manifest.json',
  '/web/icon-192.png',
  '/web/icon-512.png',
];

// Install: pre-cache app shell
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL_URLS))
      .then(() => self.skipWaiting())
  );
});

// Activate: clean old caches
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch: network-first for API, cache-first for shell
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // API requests: network only (never cache dynamic data)
  if (e.request.method !== 'GET' || url.pathname.startsWith('/ask') ||
      url.pathname.startsWith('/admin') || url.pathname.startsWith('/health') ||
      url.pathname.startsWith('/v1/')) {
    return;
  }

  // Static assets: stale-while-revalidate
  e.respondWith(
    caches.match(e.request).then(cached => {
      const fetchPromise = fetch(e.request).then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(e.request, clone));
        }
        return response;
      }).catch(() => cached);  // Network fail → use cache

      return cached || fetchPromise;
    })
  );
});
