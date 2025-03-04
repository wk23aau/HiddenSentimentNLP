let productQueue = []; // Array of product URLs to process.
let results = [];
let running = false;
let paused = false;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'start') {
    console.log('[Background] Start requested');
    running = true;
    paused = false;
    // If the product queue is empty, try to execute the content script again.
    if (productQueue.length === 0) {
      // This will inject content.js into the active tab so it can rescrape the grid.
      chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
        if (tabs.length > 0) {
          chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            files: ['content.js']
          });
        }
      });
    }
    processNextProduct();
    sendResponse({ status: 'started' });
  } else if (message.action === 'pause') {
    console.log('[Background] Pause requested');
    paused = true;
    sendResponse({ status: 'paused' });
  } else if (message.action === 'stop') {
    console.log('[Background] Stop requested');
    running = false;
    paused = false;
    // Prompt to save CSV file.
    downloadCSV(results);
    sendResponse({ status: 'stopped' });
  }
  
  // If a content script sends product URLs...
  if (message.type === 'queueProducts' && message.products) {
    productQueue = message.products;
    console.log('[Background] Received product queue:', productQueue);
  }
  
  // If a detail page returns data...
  if (message.type === 'productDetail' && message.data) {
    results.push(message.data);
    console.log('[Background] Collected product detail:', message.data);
    // Close the sending tab (if applicable)
    if (sender.tab && sender.tab.id) {
      chrome.tabs.remove(sender.tab.id);
    }
    // Continue with the next product.
    processNextProduct();
  }
  return true;
});

function processNextProduct() {
  if (!running) {
    console.log('[Background] Process stopped.');
    return;
  }
  if (paused) {
    console.log('[Background] Process is paused. Waiting...');
    setTimeout(processNextProduct, 1000);
    return;
  }
  if (productQueue.length === 0) {
    console.log('[Background] All products processed. Final results:', results);
    // Trigger CSV download when done.
    downloadCSV(results);
    running = false;
    return;
  }
  
  const nextUrl = productQueue.shift();
  console.log('[Background] Processing next product URL:', nextUrl);
  
  // Extract the ASIN from the URL.
  let asinMatch = nextUrl.match(/\/dp\/([A-Z0-9]{10})/);
  if (!asinMatch) {
    asinMatch = nextUrl.match(/\/gp\/product\/([A-Z0-9]{10})/);
  }
  if (!asinMatch) {
    console.error('[Background] Could not extract ASIN from URL:', nextUrl);
    processNextProduct();
    return;
  }
  const asin = asinMatch[1];
  const reviewsUrl = `https://www.amazon.com/product-reviews/${asin}`;
  console.log('[Background] Opening reviews page for ASIN', asin, ':', reviewsUrl);
  
  // Open the reviews page in a new, inactive tab.
  chrome.tabs.create({ url: reviewsUrl, active: false }, (tab) => {
    console.log('[Background] Created reviews tab with id:', tab.id);
    // A content script injected into the reviews page should scrape reviews and
    // send the data back (with message type "productDetail").
  });
}

function downloadCSV(data) {
  console.log('[Background] Downloading CSV with data:', data);
  // Build CSV content. (Adjust headers and fields as needed.)
  let csvContent = "url,title,review\n";
  data.forEach(row => {
    // Make sure to escape quotes if necessary.
    const line = `"${row.url}","${row.title}","${row.review}"`;
    csvContent += line + "\n";
  });
  
  // Create a Blob from the CSV content.
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  let blobUrl;
  if (typeof self !== 'undefined' && self.URL && typeof self.URL.createObjectURL === 'function') {
    blobUrl = self.URL.createObjectURL(blob);
  } else if (typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function') {
    blobUrl = URL.createObjectURL(blob);
  } else {
    console.error("[Background] Cannot create object URL for blob.");
    return;
  }
  
  chrome.downloads.download({
    url: blobUrl,
    filename: "results.csv",
    saveAs: true
  }, (downloadId) => {
    console.log('[Background] CSV download initiated with ID:', downloadId);
  });
}
