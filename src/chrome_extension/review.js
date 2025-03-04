(function () {
  let collectedReviews = [];

  // Extract review texts on the current page.
  function extractReviews() {
    const reviewElements = document.querySelectorAll('div[data-hook="review"]');
    reviewElements.forEach(el => {
      const text = el.innerText.trim();
      if (text) {
        collectedReviews.push(text);
      }
    });
    console.log("[Review] Extracted reviews count so far:", collectedReviews.length);
  }

  // Click the "Next page" button (simulate AJAX click)
  function clickNextPage() {
    const nextLink = document.querySelector('li.a-last a');
    if (nextLink && !nextLink.classList.contains('a-disabled')) {
      console.log("[Review] Clicking 'Next page' button via AJAX.");
      nextLink.click();
      setTimeout(() => {
        extractReviews();
        setTimeout(clickNextPage, 1000); // default delay for subsequent pages
      }, 1000);
    } else {
      console.log("[Review] No clickable 'Next page' button found; finishing review collection.");
      chrome.runtime.sendMessage(
        { type: "reviewData", product: window.location.href, reviews: collectedReviews },
        response => {
          console.log("[Review] Sent review data to background:", response);
        }
      );
    }
  }

  window.addEventListener("load", () => {
    setTimeout(() => {
      extractReviews();
      setTimeout(clickNextPage, 1000); // default delay after initial page load
    }, 1000);
  });
})();
