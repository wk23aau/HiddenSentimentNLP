(function waitForGrid() {
  // Adjust the selector(s) as needed – here we check for a common grid container.
  const grid = document.querySelector('ul.ProductGrid__grid__f5oba, ul.ProductGrid__asin-face-out__kKLj0');
  if (grid) {
    console.log('[Content] Product grid found.');
    // Get all <li> items with a data-csa-c-item-id attribute.
    const items = Array.from(grid.querySelectorAll('li[data-csa-c-item-id]'));
    const unique = {};
    items.forEach(item => {
      const asin = item.getAttribute('data-csa-c-item-id');
      if (asin && !unique[asin]) {
        // Find a clickable link inside the item.
        const a = item.querySelector('a[href]');
        if (a) {
          let url = a.getAttribute('href');
          if (url && !url.startsWith('http')) {
            url = window.location.origin + url;
          }
          unique[asin] = url;
        }
      }
    });
    const productUrls = Object.values(unique);
    console.log('[Content] Unique product URLs found:', productUrls);
    // Send the list of products to the background script.
    chrome.runtime.sendMessage({ type: 'queueProducts', products: productUrls });
  } else {
    console.log('[Content] Product grid not found yet – scrolling...');
    window.scrollBy(0, 500);
    // Wait 1 second and try again.
    setTimeout(waitForGrid, 1000);
  }
})();
