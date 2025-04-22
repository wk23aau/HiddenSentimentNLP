(function() {
    const styles = `
      :root {
        --primary: #2c7be5;
        --danger: #e63757;
        --success: #37b24d;
        --modal-bg: rgba(255,255,255,0.95);
        --backdrop: rgba(0,0,0,0.2);
        --border-radius: 12px;
        --shadow: 0 4px 20px rgba(0,0,0,0.1);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
  
      .sentiment-modal {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: clamp(400px, 95vw, 600px); /* Adjusted width */
        background: var(--modal-bg);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        // width: clamp(300px, 90vw, 450px);
        max-height: 80vh;
        // max-width: 35%
        overflow: hidden;
        animation: floatIn 0.5s;
        backdrop-filter: blur(10px);
        display: flex;
        flex-direction: column;
        font-family: 'Segoe UI', system-ui, sans-serif;
      }
  
      .modal-header {
        padding: 1.5rem;
        border-bottom: 1px solid #eee;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #fff, #f5f5f5);
      }
  
      .modal-content {
        padding: 1.5rem;
        overflow-y: auto;
        flex: 1;
      }
  
      .modal-footer {
        padding: 1rem;
        background: #fafafa;
        border-top: 1px solid #eee;
        display: flex;
        gap: 0.5rem;
      }
    .button-container {
    position: fixed;
    margin-top: 4rem; /* Adjusted top position */
    right: 0.5rem;
    display: flex;
    gap: 0.5rem;
    z-index: 9999;
                }
  .button-group {
  display: flex;
  gap: 10px;
  justify-content: center;
  align-items: center;
}

.btn {
  padding: 0.6rem 1.2rem;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: var(--transition);
  border: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  white-space: nowrap;
  font-size: 14px;
}

/* Specific Button Colors */
.btn-primary {
  background: linear-gradient(135deg, var(--primary), #1a68d1);
  color: white;
}

.btn-danger {
  background: linear-gradient(135deg, var(--danger), #d9254a);
  color: white;
}

.btn-success {
  background: linear-gradient(135deg, var(--success), #20c997);
  color: white;
}

.btn-outline {
  background: white;
  border: 2px solid #ddd;
  color: #333;
}

  
      .sentiment-result {
        text-align: center;
        margin: 1.5rem 0;
      }
  
      .confidence-meter {
        height: 8px;
        background: #eee;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
      }
  
      .confidence-meter span {
        display: block;
        height: 100%;
        background: linear-gradient(45deg, var(--success), #20c997);
      }
  
      @keyframes floatIn {
        from {
          transform: translateY(20px);
          opacity: 0;
        }
        to {
          transform: translateY(0);
          opacity: 1;
        }
      }
  
      .backdrop {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--backdrop);
        z-index: 9998;
        display: none;
      }
  
      .review-nav {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
      }
  
      .loader {
        border: 4px solid #f3f3f3;
        border-top: 4px solid var(--primary);
        border-radius: 50%;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
      }
  
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `;
  
    const API_URL = "http://127.0.0.1:5000/predict";
    let currentReviewIndex = 0;
    let reviews = [];
    let results = [];
  
    function getProductId(url) {
      let match = url.match(/\/dp\/([A-Z0-9]{10})/i);
      if (match && match[1]) return match[1];
      match = url.match(/\/gp\/product\/([A-Z0-9]{10})/i);
      if (match && match[1]) return match[1];
      match = url.match(/\/product-reviews\/([A-Z0-9]{10})/i);
      return match ? match[1] : null;
    }
  
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
  
    function scrapeReviews() {
      console.log("Scraping reviews from current page...");
      return Array.from(document.querySelectorAll('[data-hook="review"] span[data-hook="review-body"]'))
        .map(el => el.innerText.trim());
    }
  
    async function gatherAllReviews(filter = "all", startUrl = null) {
      let allReviews = [];
      let currentPageUrl = startUrl || window.location.href;
      
      while (currentPageUrl) {
        console.log("Fetching reviews from:", currentPageUrl);
        try {
          const response = await fetch(currentPageUrl, { credentials: 'include' });
          const html = await response.text();
          const doc = new DOMParser().parseFromString(html, "text/html");
          allReviews.push(...Array.from(doc.querySelectorAll('[data-hook="review"] span[data-hook="review-body"]'))
            .map(el => el.innerText.trim()));
          
          const nextPageLink = doc.querySelector('li.a-last a');
          currentPageUrl = nextPageLink ? new URL(nextPageLink.href).href : null;
        } catch (error) {
          console.error("Error fetching paginated reviews:", error);
          break;
        }
      }
      return allReviews;
    }
  
    async function sendReviews() {
      showLoader();
      try {
        const seeMoreLink = document.querySelector('a[data-hook="see-all-reviews-link-foot"]');
        reviews = seeMoreLink ? await gatherAllReviews() : scrapeReviews();
        
        if (!reviews.length) {
          alert("No reviews found!");
          return;
        }
  
        const response = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: reviews.join("\n") })
        });
  
        if (!response.ok) throw new Error(response.statusText);
        
        const data = await response.json();
        results = [{ review: reviews.join("\n"), result: data }];
        displayOverallResult(data);
      } catch (error) {
        console.error("API error:", error);
        alert("Analysis failed. Check console for details.");
      } finally {
        hideLoader();
      }
    }
  
    async function sendCriticalReviews() {
      showLoader();
      try {
        const productId = getProductId(window.location.href);
        if (!productId) throw new Error("Product ID not found");
        
        const startUrl = `https://www.amazon.com/product-reviews/${productId}/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=Critical&filterByStar=critical&pageNumber=1`;
        reviews = await gatherAllReviews("critical", startUrl);
        
        if (!reviews.length) {
          alert("No critical reviews found!");
          return;
        }
  
        const response = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: reviews.join("\n") })
        });
  
        if (!response.ok) throw new Error(response.statusText);
        
        const data = await response.json();
        results = [{ review: reviews.join("\n"), result: data }];
        displayOverallResult(data);
      } catch (error) {
        console.error("API error:", error);
        alert("Analysis failed. Check console for details.");
      } finally {
        hideLoader();
      }
    }
  
    async function analyzeIndividualReview(reviewText) {
      try {
        const response = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: reviewText })
        });
        return response.ok ? await response.json() : null;
      } catch (error) {
        console.error("Analysis error:", error);
        return null;
      }
    }
  
    async function analyzeEveryReview() {
      showLoader();
      results = [];
      for (const review of reviews) {
        const data = await analyzeIndividualReview(review);
        results.push({ review, result: data });
      }
      currentReviewIndex = 0;
      displayIndividualReview();
      hideLoader();
    }
  
    function displayOverallResult(data) {
      const sentiment = data.prediction === 1 ? "Positive 😊" : "Negative 😠";
      const confidence = Math.max(...data.probabilities[0]) * 100;
      
      const modal = document.createElement('div');
      modal.className = 'sentiment-modal';
      modal.innerHTML = `
        <div class="modal-header">
          <h3>Overall Sentiment Analysis</h3>
          <button class="btn btn-outline">×</button>
        </div>
        <div class="modal-content">
          <div class="sentiment-result">
            <h3>${sentiment}</h3>
            <div class="confidence-meter">
              <span style="width: ${confidence}%;"></span>
            </div>
            <p>${confidence.toFixed(2)}% Confidence</p>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary">Analyze All Reviews</button>
          <button class="btn btn-outline">Close</button>
        </div>
      `;
  
      modal.querySelector('.btn-primary').addEventListener('click', analyzeEveryReview);
      modal.querySelector('.btn-outline').addEventListener('click', () => {
        modal.remove();
        document.querySelector('.backdrop')?.remove();
      });
  
      document.body.appendChild(modal);
      document.body.insertAdjacentHTML('beforeend', '<div class="backdrop"></div>');
    }
  
    function displayIndividualReview() {
      const reviewData = results[currentReviewIndex];
      const sentiment = reviewData.result.prediction === 1 ? "Positive 😊" : "Negative 😠";
      const confidence = (Math.max(...reviewData.result.probabilities[0]) * 100).toFixed(2);
  
      const modal = document.createElement('div');
      modal.className = 'sentiment-modal';
      modal.innerHTML = `
        <div class="modal-header">
          <h3>Review ${currentReviewIndex + 1}/${results.length}</h3>
          <button class="btn btn-outline">×</button>
        </div>
        <div class="modal-content">
          <p>${reviewData.review}</p>
          <div class="sentiment-result">
            <h3>${sentiment}</h3>
            <div class="confidence-meter">
              <span style="width: ${confidence}%;"></span>
            </div>
            <p>${confidence}% Confidence</p>
          </div>
        </div>
        <div class="modal-footer">
          <div class="review-nav">
            <button class="btn btn-outline" ${currentReviewIndex === 0 ? 'disabled' : ''}>← Previous</button>
            <button class="btn btn-primary" ${currentReviewIndex === results.length - 1 ? 'disabled' : ''}>Next →</button>
          </div>
          <button class="btn btn-success">Export CSV</button>
          <button class="btn btn-danger">Close</button>
        </div>
      `;
  
      modal.querySelector('.btn-outline').addEventListener('click', () => {
        modal.remove();
        document.querySelector('.backdrop')?.remove();
      });
  
      modal.querySelector('.btn-primary').addEventListener('click', () => {
        currentReviewIndex++;
        modal.remove();
        displayIndividualReview();
      });
  
      modal.querySelector('.btn-outline:not([disabled])').addEventListener('click', () => {
        currentReviewIndex--;
        modal.remove();
        displayIndividualReview();
      });
  
      modal.querySelector('.btn-success').addEventListener('click', exportResultsToCSV);
      modal.querySelector('.btn-danger').addEventListener('click', () => {
        modal.remove();
        document.querySelector('.backdrop')?.remove();
      });
  
      document.body.appendChild(modal);
      document.body.insertAdjacentHTML('beforeend', '<div class="backdrop"></div>');
    }
  
    function exportResultsToCSV() {
      let csv = "Review,Sentiment,Confidence\n";
      results.forEach(item => {
        const sentiment = item.result.prediction === 1 ? 'Positive' : 'Negative';
        const confidence = (Math.max(...item.result.probabilities[0]) * 100).toFixed(2);
        csv += `"${item.review.replace(/"/g, '""')}",${sentiment},${confidence}%\n`;
      });
      
      const blob = new Blob([csv], { type: 'text/csv' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'reviews_sentiment_analysis.csv';
      link.click();
      URL.revokeObjectURL(link.href);
    }
  
    function addAnalyzeButton() {
      const container = document.createElement('div');
      container.className = 'button-container'; // Use the new class name
      container.style.position = 'fixed';
      container.style.top = '2rem';
      container.style.right = '2rem';
      container.style.display = 'flex';
      container.style.gap = '1rem';
      container.style.zIndex = '9999';
  
      const analyzeButton = document.createElement('button');
      analyzeButton.className = 'btn btn-primary';
      analyzeButton.textContent = 'Analyze Reviews';
      analyzeButton.addEventListener('click', sendReviews);
  
      const criticalButton = document.createElement('button');
      criticalButton.className = 'btn btn-danger';
      criticalButton.textContent = 'Analyze Critical Reviews';
      criticalButton.addEventListener('click', sendCriticalReviews);
  
      container.append(analyzeButton, criticalButton);
      document.body.appendChild(container);
    }
  
    function showLoader() {
      const loader = document.createElement('div');
      loader.className = 'loader';
      loader.id = 'globalLoader';
      document.body.appendChild(loader);
    }
  
    function hideLoader() {
      document.getElementById('globalLoader')?.remove();
    }
  
    // Initialize
    const styleSheet = document.createElement('style');
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
    window.addEventListener('load', addAnalyzeButton);
  })();