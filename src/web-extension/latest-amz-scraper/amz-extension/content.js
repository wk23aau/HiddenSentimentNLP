// content.js - Amazon Scraper (Production Ready - v2.2)
(async function() {
    console.log("Amazon Scraper Extension v2.2 loaded");

    // === CONFIGURATION ===
    const SERVER_URL = 'http://localhost:5000/api/scrape-result';
    const ASINS_URL = 'http://localhost:5000/api/asins';
    const SCRAPE_STATUS_URL = 'http://localhost:5000/api/scrape-status';
    const CONTROL_URLS = {
        start: 'http://localhost:5000/api/start-scrape',
        pause: 'http://localhost:5000/api/pause-scrape',
        resume: 'http://localhost:5000/api/resume-scrape',
        stop: 'http://localhost:5000/api/stop-scrape'
    };
    const ADD_ASINS_URL = 'http://localhost:5000/api/add-asins';

    let ASINS_QUEUE = [];
    let scrapingActive = false;
    let processingLock = false;
    let scrapeReviewsOnly = false; // Track Reviews Only mode

    const config = {
        batchSize: 20,
        totalASINs: 1000,
        processed: 0,
        domain: window.location.hostname.includes('co.uk')
            ? 'www.amazon.co.uk'
            : 'www.amazon.com'
    };

    // === UI COMPONENTS ===
    const ui = createUI();
    document.body.appendChild(ui);

    // === CORE SCRAPING LOGIC ===
    async function scrapeProduct(asin) {
        if (scrapeReviewsOnly) {
            return scrapeReviewsOnlyMode(asin); // Call Reviews Only mode function
        }

        try {
            const [productHtml, allReviews, criticalReviews, offers] = await Promise.all([
                fetchProductPage(asin),
                fetchAllReviews(asin, 'all'),
                fetchAllReviews(asin, 'critical'),
                fetchAllOffers(asin)
            ]);

            const productData = {
                asin,
                title: extractTitle(productHtml),
                brand: extractBrand(productHtml),
                categories: extractCategories(productHtml),
                pricing: extractPricing(productHtml),
                bestSellerRank: extractBestSellerRank(productHtml),
                specs: extractProductSpecs(productHtml),
                features: extractBulletPoints(productHtml),
                details: extractProductDetails(productHtml),
                buyBox: extractBuyBox(productHtml),
                availableDeals: extractAvailableDeals(productHtml),
                monthlySales: extractMonthlySales(productHtml),
                reviews: { all: allReviews, critical: criticalReviews },
                offers: offers
            };

            await sendToServer(asin, productData);

        } catch (error) {
            console.error(`[SCRAPER] Error processing ${asin}:`, error);
            await sendToServer(asin, { error: error.message });
        }
    }

    async function fetchProductPage(asin) {
        const url = `https://${config.domain}/dp/${asin}`;
        return await fetchWithRetry(url);
    }

    // === REVIEWS ONLY MODE ===
    async function scrapeReviewsOnlyMode(asin) {
        try {
            const allReviews = await fetchAllReviews(asin, 'all');
            const criticalReviews = await fetchAllReviews(asin, 'critical');

            const productData = {
                asin,
                reviews: { all: allReviews, critical: criticalReviews }
            };
            await sendToServer(asin, productData, true); // Send with partial flag
        } catch (error) {
            console.error(`[SCRAPER - Reviews Only] Error processing ${asin}:`, error);
            await sendToServer(asin, { error: error.message }, true); // Send error with partial flag
        }
    }


    // === REVIEWS EXTRACTION ===
    async function fetchAllReviews(asin, type) {
        let page = 1;
        let allReviews = [];
        let hasNext = true;

        while (hasNext) {
            try {
                const url = buildReviewUrl(asin, type, page);
                const html = await fetchWithRetry(url);
                const doc = parseHtml(html);
                const reviews = extractReviews(doc);
                allReviews.push(...reviews);
                hasNext = checkNextPage(doc);
                page++;
                await sleep(1000);
            } catch (error) {
                console.warn(`Review page ${page} failed:`, error);
                break;
            }
        }
        return allReviews;
    }

    function buildReviewUrl(asin, type, page) {
        const params = new URLSearchParams({
            pageNumber: page,
            filterByStar: type === 'critical' ? 'critical' : 'all_stars',
            ie: 'UTF8'
        });
        return `https://${config.domain}/product-reviews/${asin}?${params}`;
    }

    function extractReviews(doc) {
        return Array.from(doc.querySelectorAll('[data-hook="review"]')).map(reviewEl => {
            const ratingEl = reviewEl.querySelector('[data-hook="review-star-rating"]');
            const helpfulEl = reviewEl.querySelector('[data-hook="helpful-vote-statement"]');

            return {
                reviewer: reviewEl.querySelector('[data-hook="review-author"]')?.textContent.trim(),
                title: reviewEl.querySelector('[data-hook="review-title"]')?.textContent.trim(),
                rating: ratingEl ? parseFloat(ratingEl.textContent.match(/(\d\.?\d?)/)[0]) : null,
                date: reviewEl.querySelector('[data-hook="review-date"]')?.textContent.trim(),
                body: reviewEl.querySelector('[data-hook="review-body"]')?.textContent.trim(),
                helpfulVotes: helpfulEl ? parseHelpfulVotes(helpfulEl.textContent.trim()) : 0
            };
        });
    }

    function parseHelpfulVotes(text) {
        const match = text.match(/(\d+)\s+people/);
        return match ? parseInt(match[1], 10) : (text.includes('One') ? 1 : 0);
    }

    // === OFFERS EXTRACTION ===
    async function fetchAllOffers(asin) {
        let page = 1;
        let allOffers = [];
        let hasNext = true;

        while (hasNext) {
            try {
                const url = buildOfferUrl(asin, page);
                const html = await fetchWithRetry(url);
                const doc = parseHtml(html);
                const offers = extractOffers(doc);
                allOffers.push(...offers);
                hasNext = checkNextPage(doc);
                page++;
                await sleep(1500);
            } catch (error) {
                console.warn(`Offer page ${page} failed:`, error);
                break;
            }
        }
        return allOffers;
    }

    function buildOfferUrl(asin, page) {
        return `https://${config.domain}/gp/product/ajax/ref=aod_page_${page}` +
               `?asin=${asin}&pageno=${page}&experienceId=aodAjaxMain`;
    }

    function extractOffers(doc) {
        const offers = [];
        doc.querySelectorAll('.aod-offer').forEach(node => {
            const whole = node.querySelector('.a-price-whole')?.textContent.trim() || '';
            const frac  = node.querySelector('.a-price-fraction')?.textContent.trim() || '';
            const price = whole ? `${whole}${frac}` : node.querySelector('.a-offscreen')?.textContent.trim() || null;
            const seller = node.querySelector('.aod-offer-seller a')?.textContent.trim()
                         || node.querySelector('.aod-offer-seller .a-size-small')?.textContent.trim()
                         || null;
            const condition = node.querySelector('.aod-offer-condition')?.textContent.trim() || null;
            const shipping = node.querySelector('.aod-offer-shipping')?.textContent.trim() || null;
            offers.push({ price, seller, condition, shipping });
        });
        return offers;
    }


    // === HELPER FUNCTIONS ===
    async function fetchWithRetry(url, retries = 3) {
        for (let i = 0; i < retries; i++) {
            try {
                const response = await fetch(url);
                if (response.ok) return await response.text();
                if (response.status === 404) throw new Error('Page not found');
            } catch (error) {
                if (i === retries - 1) throw error;
                await sleep(2000 * (i + 1));
            }
        }
    }

    function parseHtml(html) {
        return new DOMParser().parseFromString(html, 'text/html');
    }

    function checkNextPage(doc) {
        return !!doc.querySelector('li.a-last:not(.a-disabled) a');
    }

    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    // === DATA EXTRACTION FUNCTIONS ===
    function extractCategories(doc) {
        return Array.from(doc.querySelectorAll('#desktop-breadcrumbs_feature_div span'))
            .filter((_, i) => i % 2 === 0)
            .map(el => el.textContent.trim());
    }

    function extractTitle(doc) {
        return doc.querySelector('#productTitle')?.textContent.trim() || 'N/A';
    }

    function extractBrand(doc) {
        const text = doc.querySelector('#bylineInfo')?.textContent.trim() || '';
        return text.split('the ')[1]?.trim() || 'N/A';
    }

    function extractRatings(doc) {
        const avg = doc.querySelector('#acrPopover span.a-size-base.a-color-base')?.textContent.trim() || 'N/A';
        const total = doc.querySelector('#acrCustomerReviewText')?.textContent.replace(/\D/g, '') || '0';
        return { average: avg, total };
    }

    function extractMonthlySales(doc) {
        const el = doc.querySelector('#socialProofingAsinFaceout_feature_div span');
        return el?.textContent.match(/^\d+[Kk]\+/)?.[0] || 'N/A';
    }

    function extractPricing(doc) {
        const c = doc.querySelector('#corePriceDisplay_desktop_feature_div');
        return {
            current: c?.querySelector('.a-price-whole')?.textContent.trim() || 'N/A',
            savings: c?.querySelector('.savingPriceOverride')?.textContent.trim() || 'N/A',
            listPrice: c?.querySelector('.basisPrice .a-offscreen')?.textContent.trim() || 'N/A'
        };
    }

    function extractBestSellerRank(doc) {
        return doc.querySelector('#zeitgeistBadge_feature_div span.cat-link')?.textContent.trim() || 'N/A';
    }

    function extractProductSpecs(doc) {
        const specs = {};
        doc.querySelectorAll('#poExpander tr.a-spacing-small').forEach(row => {
            const key = row.querySelector('td:first-child span.a-text-bold')?.textContent.trim();
            const val = row.querySelector('td:last-child span.po-break-word')?.textContent.trim();
            if (key && val) specs[key] = val;
        });
        return specs;
    }

    function extractBulletPoints(doc) {
        return Array.from(doc.querySelectorAll('#featurebullets_feature_div span.a-list-item'))
            .map(el => el.textContent.trim())
            .filter(Boolean);
    }

    function extractProductDetails(doc) {
        const details = {};
        const primary = doc.querySelector('#productDetails_feature_div');
        if (primary) {
            primary.querySelectorAll('table.a-keyvalue tr').forEach(r => {
                const k = r.querySelector('th')?.textContent.trim().replace(/[:\s]+$/, '');
                const v = r.querySelector('td')?.textContent.trim();
                if (k && v) details[k] = v;
            });
        } else {
            doc.querySelectorAll('.content-grid-row-wrapper table.a-bordered tr').forEach(r => {
                const key = r.querySelector('td:first-child')?.textContent.trim();
                const val = r.querySelector('td:last-child')?.textContent.trim();
                if (key && val) details[key] = val;
            });
        }
        return details;
    }

    function extractBuyBox(doc) {
        return {
            availability: doc.querySelector('#availability span')?.textContent.trim() || 'N/A',
            price: doc.querySelector('.a-offscreen')?.textContent.trim() || 'N/A'
        };
    }

    function extractAvailableDeals(doc) {
        return doc.querySelector('#dealBadgeSupportingText span')?.textContent.trim() || 'N/A';
    }


    // === SERVER COMMUNICATION ===
    async function sendToServer(asin, data, partial = false) { // Added partial parameter
        try {
            const payload = { asin, productData: data };
            if (partial) {
                payload.partial = true; // Add partial flag to payload
            }
            const response = await fetch(SERVER_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            console.log(`[SERVER] Data sent for ${asin} (Partial: ${partial})`);
        } catch (error) {
            console.error(`[SERVER] Failed for ${asin} (Partial: ${partial}):`, error);
        }
    }

    async function fetchASINsFromServer() {
        try {
            const response = await fetch(ASINS_URL);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const asins = await response.json();
            ASINS_QUEUE.push(...asins);
            console.log(`[ASINS] Fetched ${asins.length} ASINs from server.`);
        } catch (error) {
            console.error("[ASINS] Error fetching ASINs:", error);
            updateStatusDisplay('Error fetching ASINs');
        }
    }


    // === PROCESSING LOOP ===
    async function processQueue() {
        if (processingLock || !scrapingActive) return;

        processingLock = true;
        await fetchASINsFromServer(); // Fetch ASINs at the beginning of each processing cycle

        if (ASINS_QUEUE.length === 0 && scrapingActive) {
            updateStatusDisplay('Waiting for ASINs...');
            processingLock = false;
            setTimeout(processQueue, 5000); // Wait and check again
            return;
        }

        const batchSize = Math.min(
            config.batchSize,
            ASINS_QUEUE.length,
            config.totalASINs - config.processed
        );

        if (batchSize <= 0 && scrapingActive) {
            processingLock = false;
            if (config.processed >= config.totalASINs) {
                stopScraping(); // Stop if total ASINs processed
            } else {
                setTimeout(processQueue, 5000); // Wait for more ASINs if limit not reached
            }
            return;
        }


        const batch = ASINS_QUEUE.splice(0, batchSize);
        await Promise.all(batch.map(scrapeProduct));
        config.processed += batch.length;
        processingLock = false;
        updateStatusDisplay(`Running (${config.processed}/${config.totalASINs})`);

        if (config.processed < config.totalASINs && scrapingActive) {
            setTimeout(processQueue, 1000);
        } else {
            stopScraping();
        }
    }


    // === UI & CONTROLS ===
    function createUI() {
        const container = document.createElement('div');
        container.id = 'scraper-ui';
        container.style.cssText = `
            position: fixed; top: 20px; left: 20px;
            background: #fff; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); z-index: 1000;
            font-family: Arial, sans-serif;
        `;

        const buttons = {
            start: createButton('Start', () => openConfigModal()),
            pause: createButton('Pause', pauseScraping),
            resume: createButton('Resume', resumeScraping),
            stop: createButton('Stop', stopScraping),
            asin: createButton('Scrape ASINs', () => scrapeASINsPrompt())
        };

        const reviewsOnlySwitch = document.createElement('label'); // Reviews Only Switch
        reviewsOnlySwitch.style.cssText = `
            margin-left: 20px;
            display: inline-flex;
            align-items: center;
            font-size: 14px;
        `;
        reviewsOnlySwitch.innerHTML = `
            <input type="checkbox" id="reviewsOnlyToggle">
            <span style="margin-left: 5px">Reviews Only</span>
        `;


        const status = document.createElement('span');
        status.id = 'scraper-status';
        status.textContent = 'Status: Idle';
        status.style.marginLeft = '10px';

        Object.values(buttons).forEach(btn => container.appendChild(btn));
        container.appendChild(reviewsOnlySwitch); // Add Reviews Only Switch
        container.appendChild(status);
        return container;
    }

    function createButton(text, onClick) {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.onclick = onClick;
        btn.style.cssText = `
            padding: 8px 12px; margin-right: 5px;
            border: 1px solid #ccc; border-radius: 5px;
            background-color: #eee; cursor: pointer;
        `;
        return btn;
    }

    function updateStatusDisplay(statusText) {
        const statusDisplay = document.getElementById('scraper-status');
        if (statusDisplay) statusDisplay.textContent = `Status: ${statusText}`;
    }

    // === CONFIGURATION MODAL ===
    function openConfigModal() {
        const modal = document.createElement('div');
        modal.id = 'config-modal';
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center;
        `;

        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: #fff; padding: 20px; border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        `;

        modalContent.innerHTML = `
            <h2>Scraper Configuration</h2>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px;">Batch Size (1-50):</label>
                <input type="number" id="batch-size" value="${config.batchSize}" min="1" max="50" style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ccc;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px;">Total ASINs to Process (1-5000):</label>
                <input type="number" id="total-asins" value="${config.totalASINs}" min="1" max="5000" style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ccc;">
            </div>
            <div style="text-align: right;">
                <button id="start-btn" style="padding: 10px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">Start</button>
                <button id="cancel-btn" style="padding: 10px 15px; background-color: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px;">Cancel</button>
            </div>
        `;

        modalContent.querySelector('#start-btn').onclick = () => {
            const batchSizeInput = modalContent.querySelector('#batch-size');
            const totalAsinsInput = modalContent.querySelector('#total-asins');
            config.batchSize = parseInt(batchSizeInput.value, 10);
            config.totalASINs = parseInt(totalAsinsInput.value, 10);
            config.processed = 0; // Reset processed count on new start
            updateStatusDisplay(`Starting (Batch: ${config.batchSize}, Total: ${config.totalASINs})`);
            startScraping();
            document.body.removeChild(modal);
        };

        modalContent.querySelector('#cancel-btn').onclick = () => {
            document.body.removeChild(modal);
        };

        modal.appendChild(modalContent);
        document.body.appendChild(modal);
    }


    // === CONTROL FUNCTIONS ===
    function startScraping() {
        fetch(CONTROL_URLS.start, { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    scrapingActive = true;
                    processQueue();
                    updateStatusDisplay(`Running (0/${config.totalASINs})`);
                } else {
                    updateStatusDisplay('Error starting');
                }
            })
            .catch(() => updateStatusDisplay('Error starting'));
    }

    function pauseScraping() {
        fetch(CONTROL_URLS.pause, { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    scrapingActive = false;
                    updateStatusDisplay('Paused');
                } else {
                    updateStatusDisplay('Error pausing');
                }
            })
            .catch(() => updateStatusDisplay('Error pausing'));
    }

    function resumeScraping() {
        fetch(CONTROL_URLS.resume, { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    scrapingActive = true;
                    processQueue();
                    updateStatusDisplay(`Running (${config.processed}/${config.totalASINs})`);
                } else {
                    updateStatusDisplay('Error resuming');
                }
            })
            .catch(() => updateStatusDisplay('Error resuming'));
    }

    function stopScraping() {
        fetch(CONTROL_URLS.stop, { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    scrapingActive = false;
                    updateStatusDisplay('Stopped');
                } else {
                    updateStatusDisplay('Error stopping');
                }
            })
            .catch(() => updateStatusDisplay('Error stopping'));
    }

    // === ASIN SCRAPING FROM PAGE ===
    function scrapeASINsPrompt() {
        const pagesInput = prompt("Enter number of pages to scrape ASINs from (default 1):", "1");
        const pagesToScrape = pagesInput ? parseInt(pagesInput, 10) : 1;
        if (isNaN(pagesToScrape) || pagesToScrape < 1) {
            alert("Invalid number of pages.");
            return;
        }
        scrapeASINPages(pagesToScrape);
    }

    async function scrapeASINPages(maxPages) {
        let currentPage = 1;
        let currentUrl = window.location.href;
        updateStatusDisplay(`Scraping ASINs (Page 1/${maxPages})...`);

        while (currentPage <= maxPages) {
            try {
                const response = await fetch(currentUrl);
                if (!response.ok) {
                    console.error(`Failed to fetch page ${currentPage}: ${response.status}`);
                    updateStatusDisplay(`ASIN scrape failed at page ${currentPage}`);
                    break;
                }
                const html = await response.text();
                const doc = parseHtml(html);
                const pageASINs = getASINs(doc);
                if (pageASINs.length > 0) {
                    await sendASINsToServer(pageASINs);
                    console.log(`[ASINS] Page ${currentPage}: Sent ${pageASINs.length} ASINs to server.`);
                } else {
                    console.log(`[ASINS] Page ${currentPage}: No ASINs found.`);
                }

                if (currentPage < maxPages) {
                    const nextBtn = doc.querySelector('a.s-pagination-next:not(.s-pagination-disabled)');
                    if (nextBtn) {
                        currentUrl = new URL(nextBtn.href, location.origin).href;
                        currentPage++;
                        updateStatusDisplay(`Scraping ASINs (Page ${currentPage}/${maxPages})...`);
                        await sleep(1500);
                    } else {
                        console.log("[ASINS] No more 'Next' page button found.");
                        break;
                    }
                } else {
                    break; // Max pages reached
                }

            } catch (error) {
                console.error(`[ASINS] Error scraping page ${currentPage}:`, error);
                updateStatusDisplay(`ASIN scrape error at page ${currentPage}`);
                break;
            }
        }
        updateStatusDisplay('ASIN Scrape Complete.');
        console.log("[ASINS] ASIN scraping process completed.");
    }


    function getASINs(doc) {
        const asinElements = doc.querySelectorAll('[data-asin]');
        return Array.from(asinElements)
            .map(element => element.dataset.asin)
            .filter(asin => asin && asin.trim() !== "");
    }

    async function sendASINsToServer(asins) {
        if (asins.length === 0) return;
        try {
            const response = await fetch(ADD_ASINS_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ asins })
            });
            if (!response.ok) {
                console.error("[ASINS] Failed to send ASINs to server:", response.status);
            }
        } catch (error) {
            console.error("[ASINS] Error sending ASINs:", error);
        }
    }


    // === INITIALIZATION ===
    async function getScrapeStatusFromServer() {
        try {
            const response = await fetch(SCRAPE_STATUS_URL);
            if (!response.ok) return 'stopped';
            const { status } = await response.json();
            return status;
        } catch (error) {
            return 'stopped';
        }
    }

    async function init() {
        const initialStatus = await getScrapeStatusFromServer();
        scrapingActive = ['running', 'resumed'].includes(initialStatus);
        updateStatusDisplay(scrapingActive ? `Running (${config.processed}/${config.totalASINs})` : 'Idle');

        // Initialize Reviews Only Toggle Event Listener
        const reviewsOnlyToggle = document.getElementById('reviewsOnlyToggle');
        if (reviewsOnlyToggle) {
            reviewsOnlyToggle.addEventListener('change', (e) => {
                scrapeReviewsOnly = e.target.checked;
                console.log(`Reviews Only mode ${scrapeReviewsOnly ? 'enabled' : 'disabled'}`);
            });
        }

        if (scrapingActive) processQueue();
    }

    init();

})();