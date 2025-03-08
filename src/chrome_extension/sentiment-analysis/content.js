(function() {
  const API_URL = "http://127.0.0.1:5000/predict";
  let currentReviewIndex = 0;
  let reviews = [];
  let results = [];

  // Helper function to extract product ID from various Amazon URL formats.
  function getProductId(url) {
      // Try matching /dp/{id}
      let match = url.match(/\/dp\/([A-Z0-9]{10})/i);
      if (match && match[1]) return match[1];

      // Try matching /gp\/product\/{id}
      match = url.match(/\/gp\/product\/([A-Z0-9]{10})/i);
      if (match && match[1]) return match[1];

      // Try matching /product-reviews/{id}
      match = url.match(/\/product-reviews\/([A-Z0-9]{10})/i);
      if (match && match[1]) return match[1];

      return null;
  }

  // Helper: Wait until the reviews list updates (i.e. the count changes) or until timeout.
  function waitForReviewsUpdate(previousCount, timeout = 5000) {
      return new Promise(resolve => {
          const interval = setInterval(() => {
              const currentCount = document.querySelectorAll('[data-hook="review"] span[data-hook="review-body"]').length;
              if (currentCount !== previousCount) {
                  clearInterval(interval);
                  resolve();
              }
          }, 500);
          setTimeout(() => {
              clearInterval(interval);
              resolve();
          }, timeout);
      });
  }

  // Scrape reviews from the current page.
  function scrapeReviews() {
      console.log("Scraping reviews from current page...");
      const pageReviews = [];
      document.querySelectorAll('[data-hook="review"] span[data-hook="review-body"]').forEach(el => {
          pageReviews.push(el.innerText.trim());
      });
      console.log("Reviews scraped:", pageReviews);
      return pageReviews;
  }

  /**
   * Gather all reviews from paginated pages.
   * Optionally accepts a starting URL (if provided, it overrides the default from the "See more reviews" link).
   */
  async function gatherAllReviews(filter = "all", startUrl = null) {
      let allReviews = [];
      let currentPageUrl = startUrl;
      if (!currentPageUrl) {
          const seeMoreLink = document.querySelector('a[data-hook="see-all-reviews-link-foot"]');
          if (seeMoreLink) {
              currentPageUrl = seeMoreLink.getAttribute("href");
              if (currentPageUrl && !currentPageUrl.startsWith("http")) {
                  currentPageUrl = window.location.origin + currentPageUrl;
              }
          } else {
              // No pagination button found: fallback to current page scraping.
              return scrapeReviews();
          }
      }
      while (currentPageUrl) {
          console.log("Fetching reviews from:", currentPageUrl);
          try {
              const response = await fetch(currentPageUrl, { credentials: 'include' });
              const text = await response.text();
              const parser = new DOMParser();
              const doc = parser.parseFromString(text, "text/html");
              // Extract reviews from this page.
              doc.querySelectorAll('[data-hook="review"] span[data-hook="review-body"]').forEach(el => {
                  allReviews.push(el.innerText.trim());
              });
              console.log(`Total reviews fetched so far: ${allReviews.length}`);
              // Look for the "Next page" link.
              const nextPageLinkElement = doc.querySelector('li.a-last a');
              if (nextPageLinkElement) {
                  let nextPageUrl = nextPageLinkElement.getAttribute("href");
                  if (nextPageUrl && !nextPageUrl.startsWith("http")) {
                      nextPageUrl = window.location.origin + nextPageUrl;
                  }
                  currentPageUrl = nextPageUrl;
              } else {
                  currentPageUrl = null;
              }
          } catch (error) {
              console.error("Error fetching paginated reviews:", error);
              break;
          }
      }
      return allReviews;
  }

  // Send all reviews (all-stars) for overall analysis.
  async function sendReviews() {
      let allReviews = [];
      const seeMoreLink = document.querySelector('a[data-hook="see-all-reviews-link-foot"]');
      if (seeMoreLink) {
          console.log("Pagination detected. Gathering all reviews...");
          allReviews = await gatherAllReviews("all");
      } else {
          allReviews = scrapeReviews();
      }
      if (allReviews.length === 0) {
          alert("No reviews found on this page!");
          return;
      }
      reviews = allReviews;
      try {
          console.log("Sending reviews to API...");
          const response = await fetch(API_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ "text": allReviews.join("\n") })
          });
          if (!response.ok) throw new Error(response.statusText);
          const data = await response.json();
          console.log("Received API response:", data);
          results = [{ review: allReviews.join("\n"), result: data }];
          displayOverallResult(data);
      } catch (error) {
          console.error("API error:", error);
          alert("Failed to analyze reviews.");
      }
  }

  // Send only critical reviews for analysis by constructing the URL manually.
  async function sendCriticalReviews() {
      // Use current window URL to detect product ID.
      const currentUrl = window.location.href;
      const productId = getProductId(currentUrl);
      if (!productId) {
          alert("Product ID not found in URL!");
          return;
      }
      // Construct the critical reviews URL.
      const startUrl = `https://www.amazon.com/product-reviews/${productId}/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=Critical&filterByStar=critical&pageNumber=1`;
      console.log("Constructed critical reviews URL:", startUrl);
      // Gather reviews starting from the constructed URL.
      const allReviews = await gatherAllReviews("critical", startUrl);
      if (allReviews.length === 0) {
          alert("No critical reviews found on this page!");
          return;
      }
      reviews = allReviews;
      try {
          console.log("Sending critical reviews to API...");
          const response = await fetch(API_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ "text": allReviews.join("\n") })
          });
          if (!response.ok) throw new Error(response.statusText);
          const data = await response.json();
          console.log("Received API response:", data);
          results = [{ review: allReviews.join("\n"), result: data }];
          displayOverallResult(data);
      } catch (error) {
          console.error("API error:", error);
          alert("Failed to analyze reviews.");
      }
  }

  // Analyze an individual review.
  async function analyzeIndividualReview(reviewText) {
      console.log("Analyzing individual review:", reviewText);
      const response = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ "text": reviewText })
      });
      return response.ok ? await response.json() : null;
  }

  async function analyzeEveryReview() {
      results = [];
      console.log("Analyzing every individual review...");
      for (const review of reviews) {
          const data = await analyzeIndividualReview(review);
          results.push({ review, result: data });
      }
      currentReviewIndex = 0;
      displayIndividualReview();
  }

  // Display overall sentiment analysis result.
  function displayOverallResult(data) {
      const sentiment = data.prediction === 1 ? "Positive 😊" : "Negative 😞";
      const confidence = (Math.max(...data.probabilities[0]) * 100).toFixed(2);
      const container = document.createElement("div");
      Object.assign(container.style, {
          position: "fixed",
          bottom: "15px",
          right: "15px",
          backgroundColor: "#fff",
          padding: "15px",
          borderRadius: "10px",
          boxShadow: "0 0 10px rgba(0,0,0,0.3)",
          zIndex: "9999",
          width: "350px",
          fontFamily: "Arial"
      });
      container.innerHTML = `
          <h4>Overall Sentiment: ${sentiment}</h4>
          <p><strong>Confidence:</strong> ${confidence}%</p>
          <button id="analyzeAll">Analyze Every Review Individually</button>
          <button id="closeButton">Close</button>
      `;
      document.body.appendChild(container);
      document.getElementById("analyzeAll").onclick = () => {
          container.remove();
          analyzeEveryReview();
      };
      document.getElementById("closeButton").onclick = () => container.remove();
  }

  // Display individual review sentiment results with navigation.
  function displayIndividualReview() {
      const existingContainer = document.getElementById("individualReviewContainer");
      if (existingContainer) existingContainer.remove();
      const reviewData = results[currentReviewIndex];
      const sentiment = reviewData.result.prediction === 1 ? "Positive 😊" : "Negative 😞";
      const confidence = (Math.max(...reviewData.result.probabilities[0]) * 100).toFixed(2);
      const reviewContainer = document.createElement("div");
      reviewContainer.id = "individualReviewContainer";
      Object.assign(reviewContainer.style, {
          position: "fixed",
          bottom: "15px",
          right: "15px",
          backgroundColor: "#fff",
          padding: "15px",
          borderRadius: "10px",
          boxShadow: "0 0 12px rgba(0,0,0,0.2)",
          width: "400px",
          maxHeight: "300px",
          overflowY: "auto",
          zIndex: "99999"
      });
      reviewContainer.innerHTML = `
          <h4>Review ${currentReviewIndex + 1}/${results.length}</h4>
          <p>${reviewData.review}</p>
          <p><strong>Sentiment:</strong> ${sentiment}</p>
          <p><strong>Confidence:</strong> ${confidence}%</p>
          <button id="prevReview">Previous</button>
          <button id="nextReview">Next</button>
          <button id="exportResults">Export CSV</button>
          <button id="closeIndividual">Close</button>
      `;
      document.body.appendChild(reviewContainer);
      document.getElementById("closeIndividual").onclick = () => reviewContainer.remove();
      document.getElementById("prevReview").onclick = () => {
          if (currentReviewIndex > 0) {
              currentReviewIndex--;
              reviewContainer.remove();
              displayIndividualReview();
          }
      };
      document.getElementById("nextReview").onclick = () => {
          if (currentReviewIndex < results.length - 1) {
              currentReviewIndex++;
              reviewContainer.remove();
              displayIndividualReview();
          }
      };
      document.getElementById("exportResults").onclick = exportResultsToCSV;
  }

  // Export the sentiment analysis results to CSV.
  function exportResultsToCSV() {
      let csv = "Review,Sentiment,Confidence\n";
      results.forEach(item => {
          const sentimentText = item.result.prediction === 1 ? 'Positive' : 'Negative';
          const confidenceValue = (Math.max(...item.result.probabilities[0]) * 100).toFixed(2);
          const row = `"${item.review.replace(/"/g, '""')}",${sentimentText},${confidenceValue}%`;
          csv += row + "\n";
      });
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "reviews_sentiment_analysis.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
  }

  // Add floating buttons to trigger the analysis process.
  function addAnalyzeButton() {
      const container = document.createElement("div");
      Object.assign(container.style, {
          position: "fixed",
          top: "10px",
          right: "10px",
          zIndex: "99999",
          display: "flex",
          flexDirection: "column",
          gap: "5px"
      });
      const analyzeButton = document.createElement("button");
      analyzeButton.innerText = "Analyze Reviews";
      Object.assign(analyzeButton.style, {
          padding: "8px 12px",
          backgroundColor: "#007bff",
          color: "#fff",
          borderRadius: "5px",
          cursor: "pointer"
      });
      analyzeButton.onclick = sendReviews;
      container.appendChild(analyzeButton);
      const criticalButton = document.createElement("button");
      criticalButton.innerText = "Analyze Critical Reviews";
      Object.assign(criticalButton.style, {
          padding: "8px 12px",
          backgroundColor: "#dc3545",
          color: "#fff",
          borderRadius: "5px",
          cursor: "pointer"
      });
      criticalButton.onclick = sendCriticalReviews;
      container.appendChild(criticalButton);
      document.body.appendChild(container);
  }

  window.addEventListener("load", addAnalyzeButton);
})();
