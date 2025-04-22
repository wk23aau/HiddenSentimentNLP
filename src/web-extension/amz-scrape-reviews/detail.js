// detail.js
(async function() {
  console.log('[Detail] Script started on reviews page:', window.location.href);

  // Utility: Retrieve the accumulated reviews from chrome.storage.local.
  async function getStoredReviews() {
    return new Promise(resolve => {
      chrome.storage.local.get("reviewsAccumulated", (result) => {
        resolve(result.reviewsAccumulated || []);
      });
    });
  }

  // Utility: Store the accumulated reviews.
  async function storeReviews(reviews) {
    return new Promise(resolve => {
      chrome.storage.local.set({ reviewsAccumulated: reviews }, resolve);
    });
  }

  // Wait for the page/reviews to load (adjust delay as needed).
  await new Promise(resolve => setTimeout(resolve, 10));

  // Scrape reviews on the current page.
  function scrapeReviews() {
    const reviewElements = document.querySelectorAll('div[data-hook="review"]');
    console.log(`[Detail] Found ${reviewElements.length} review elements on this page.`);
    const reviews = [];
    reviewElements.forEach(el => {
      const bodyEl = el.querySelector("span[data-hook='review-body']");
      const ratingEl = el.querySelector("i[data-hook='review-star-rating'], i[data-hook='cmps-review-star-rating']");
      const reviewText = bodyEl ? bodyEl.innerText.trim() : "";
      const reviewRating = ratingEl ? ratingEl.innerText.trim() : "";
      reviews.push({ review: reviewText, rating: reviewRating });
    });
    return reviews;
  }

  // Get reviews from this page and accumulate them.
  let currentReviews = scrapeReviews();
  let storedReviews = await getStoredReviews();
  let allReviews = storedReviews.concat(currentReviews);
  await storeReviews(allReviews);
  console.log("[Detail] Total reviews accumulated so far:", allReviews.length);

  // Check if a "Next page" button exists and is enabled.
  // (Amazon's pagination uses an <li class="a-last">.
  // When disabled, the <li> gets the class "a-disabled".)
  const nextPageLI = document.querySelector('li.a-last');
  if (nextPageLI && !nextPageLI.classList.contains("a-disabled")) {
    const nextLink = nextPageLI.querySelector("a");
    if (nextLink) {
      console.log("[Detail] Next page found. Navigating to:", nextLink.href);
      // Wait a bit before navigating so the user can see progress.
      setTimeout(() => {
        window.location.href = nextLink.href;
      }, 2000);
    } else {
      console.log("[Detail] Next page list item found but no link exists.");
      finalize();
    }
  } else {
    console.log("[Detail] No enabled next page button found. Finalizing reviews.");
    finalize();
  }

  // When pagination is done, send the final results to the background script.
  function finalize() {
    // Optionally, you could also clear stored reviews here.
    console.log("[Detail] Final accumulated reviews:", allReviews.length);
    chrome.runtime.sendMessage(
      { type: "productDetail", data: { url: window.location.href, reviews: allReviews } },
      response => {
        console.log('[Detail] Sent final product detail, response:', response);
      }
    );
  }
})();
