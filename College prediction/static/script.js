/**
 * IIT Admission Prediction Engine — Client-Side Logic
 * =====================================================
 * Handles API calls, form validation, dynamic rendering, filtering, and sorting.
 * IIT-only version: no state/quota routing needed.
 */

(function () {
    "use strict";

    // ── DOM References ──
    const form = document.getElementById("predict-form");
    const rankInput = document.getElementById("rank-input");
    const categorySelect = document.getElementById("category-select");
    const genderSelect = document.getElementById("gender-select");
    const predictBtn = document.getElementById("predict-btn");

    const loader = document.getElementById("loader");
    const summarySection = document.getElementById("summary-section");
    const resultsSection = document.getElementById("results-section");
    const noResults = document.getElementById("no-results");
    const queryBanner = document.getElementById("query-banner");
    const errorToast = document.getElementById("error-toast");
    const resultsBody = document.getElementById("results-body");
    const searchInput = document.getElementById("search-input");

    // ── State ──
    let allPredictions = [];
    let filteredPredictions = [];
    let activeFilters = { tier: "all", search: "" };
    let sortConfig = { key: null, ascending: true };

    // ── Initialize ──
    document.addEventListener("DOMContentLoaded", () => {
        loadMetadata();
        setupEventListeners();
    });

    // ══════════════════════════════════════════
    // API Calls
    // ══════════════════════════════════════════

    async function loadMetadata() {
        try {
            const res = await fetch("/api/metadata");
            if (!res.ok) throw new Error("Failed to load metadata");
            const data = await res.json();

            populateSelect(categorySelect, data.categories, formatCategory);
            populateSelect(genderSelect, data.genders, formatGender);
        } catch (err) {
            showError("Could not load form options. Is the server running?");
            console.error(err);
        }
    }

    async function submitPrediction(rank, category, gender) {
        showLoader(true);
        hideResults();

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    rank: parseInt(rank),
                    category,
                    gender,
                }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Prediction failed");
            }

            const data = await res.json();
            showLoader(false);

            // Update query banner
            document.getElementById("query-rank").textContent = parseInt(rank).toLocaleString();
            document.getElementById("query-category").textContent = formatCategory(category);
            document.getElementById("query-gender").textContent = formatGender(gender);
            queryBanner.classList.add("active");

            if (data.predictions.length === 0) {
                noResults.classList.add("active");
                return;
            }

            allPredictions = data.predictions;
            filteredPredictions = [...allPredictions];

            renderSummary(data.summary);
            renderResults();
            resetFilters();

        } catch (err) {
            showLoader(false);
            showError(err.message);
            console.error(err);
        }
    }

    // ══════════════════════════════════════════
    // Rendering
    // ══════════════════════════════════════════

    function renderSummary(summary) {
        animateCounter("stat-total", summary.total);
        animateCounter("stat-safe", summary.safe);
        animateCounter("stat-moderate", summary.moderate);
        animateCounter("stat-ambitious", summary.ambitious);
        summarySection.classList.add("active");
    }

    function renderResults() {
        resultsBody.innerHTML = "";

        if (filteredPredictions.length === 0) {
            noResults.classList.add("active");
            resultsSection.classList.remove("active");
            return;
        }

        noResults.classList.remove("active");
        resultsSection.classList.add("active");

        filteredPredictions.forEach((p, idx) => {
            const tr = document.createElement("tr");
            tr.dataset.tierCode = p.risk_tier_code;

            // Extract short IIT name (e.g., "IIT Bombay" from full name)
            const shortName = p.institute
                .replace("Indian Institute of Technology", "IIT")
                .replace("(", "")
                .replace(")", "")
                .trim();

            tr.innerHTML = `
                <td class="rank-display">${idx + 1}</td>
                <td>
                    <div class="institute-name">${escapeHtml(shortName)}</div>
                    <div class="program-name">${escapeHtml(p.program)}</div>
                </td>
                <td>${tierBadge(p.risk_tier, p.risk_tier_code)}</td>
                <td class="rank-display">${p.avg_closing_rank.toLocaleString()}</td>
                <td class="rank-display">${p.min_closing_rank.toLocaleString()} – ${p.max_closing_rank.toLocaleString()}</td>
                <td><span class="round-badge">${p.earliest_round}</span></td>
                <td class="rank-display">${p.years_of_data}</td>
            `;

            resultsBody.appendChild(tr);
        });
    }

    function tierBadge(label, code) {
        const cls = code === 1 ? "tier-safe" : code === 2 ? "tier-moderate" : "tier-ambitious";
        const short = code === 1 ? "Safe" : code === 2 ? "Realistic" : "Ambitious";
        return `<span class="tier-badge ${cls}"><span class="tier-dot"></span>${short}</span>`;
    }

    // ══════════════════════════════════════════
    // Filtering
    // ══════════════════════════════════════════

    function applyFilters() {
        filteredPredictions = allPredictions.filter((p) => {
            // Tier filter
            if (activeFilters.tier !== "all" && p.risk_tier_code !== parseInt(activeFilters.tier)) {
                return false;
            }
            // Search filter
            if (activeFilters.search) {
                const q = activeFilters.search.toLowerCase();
                if (
                    !p.institute.toLowerCase().includes(q) &&
                    !p.program.toLowerCase().includes(q)
                ) {
                    return false;
                }
            }
            return true;
        });

        // Re-apply current sort
        if (sortConfig.key) {
            sortResults(sortConfig.key, false);
        }

        renderResults();
    }

    function resetFilters() {
        activeFilters = { tier: "all", search: "" };
        searchInput.value = "";

        // Reset chip states
        document.querySelectorAll(".filter-chip[data-filter]").forEach((chip) => {
            chip.classList.toggle("active", chip.dataset.filter === "all");
        });
    }

    // ══════════════════════════════════════════
    // Sorting
    // ══════════════════════════════════════════

    function sortResults(key, toggleDirection = true) {
        if (toggleDirection) {
            if (sortConfig.key === key) {
                sortConfig.ascending = !sortConfig.ascending;
            } else {
                sortConfig.key = key;
                sortConfig.ascending = true;
            }
        }

        const dir = sortConfig.ascending ? 1 : -1;

        filteredPredictions.sort((a, b) => {
            let va, vb;
            switch (key) {
                case "institute":
                    va = a.institute.toLowerCase();
                    vb = b.institute.toLowerCase();
                    return va < vb ? -dir : va > vb ? dir : 0;
                case "tier":
                    return (a.risk_tier_code - b.risk_tier_code) * dir;
                case "avg_closing":
                    return (a.avg_closing_rank - b.avg_closing_rank) * dir;
                case "range":
                    return (a.min_closing_rank - b.min_closing_rank) * dir;
                case "round":
                    return (a.earliest_round - b.earliest_round) * dir;
                case "years":
                    return (a.years_of_data - b.years_of_data) * dir;
                default:
                    return 0;
            }
        });

        // Update header sort indicators
        document.querySelectorAll(".results-table th").forEach((th) => {
            th.classList.toggle("sorted", th.dataset.sort === key);
            const icon = th.querySelector(".sort-icon");
            if (icon) {
                icon.textContent = th.dataset.sort === key
                    ? (sortConfig.ascending ? "↑" : "↓")
                    : "↕";
            }
        });
    }

    // ══════════════════════════════════════════
    // Event Listeners
    // ══════════════════════════════════════════

    function setupEventListeners() {
        // Form submission
        form.addEventListener("submit", (e) => {
            e.preventDefault();

            const rank = rankInput.value.trim();
            const category = categorySelect.value;
            const gender = genderSelect.value;

            if (!rank || !category || !gender) {
                showError("Please fill in all fields.");
                return;
            }

            if (parseInt(rank) <= 0) {
                showError("Rank must be a positive number.");
                return;
            }

            submitPrediction(rank, category, gender);
        });

        // Tier filter chips
        document.querySelectorAll(".filter-chip[data-filter]").forEach((chip) => {
            chip.addEventListener("click", () => {
                activeFilters.tier = chip.dataset.filter;
                document.querySelectorAll(".filter-chip[data-filter]").forEach((c) => {
                    c.classList.toggle("active", c.dataset.filter === activeFilters.tier);
                });
                applyFilters();
            });
        });

        // Search input
        let searchTimeout;
        searchInput.addEventListener("input", () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                activeFilters.search = searchInput.value.trim();
                applyFilters();
            }, 200);
        });

        // Column sorting
        document.querySelectorAll(".results-table th[data-sort]").forEach((th) => {
            th.addEventListener("click", () => {
                sortResults(th.dataset.sort);
                renderResults();
            });
        });
    }

    // ══════════════════════════════════════════
    // Utilities
    // ══════════════════════════════════════════

    function populateSelect(select, items, formatter) {
        items.forEach((item) => {
            const option = document.createElement("option");
            option.value = item;
            option.textContent = formatter ? formatter(item) : item;
            select.appendChild(option);
        });
    }

    function formatCategory(cat) {
        const map = {
            "OPEN": "General (OPEN)",
            "OBC-NCL": "OBC-NCL",
            "SC": "SC",
            "ST": "ST",
            "EWS": "EWS",
            "OPEN (PwD)": "General (PwD)",
            "OBC-NCL (PwD)": "OBC-NCL (PwD)",
            "SC (PwD)": "SC (PwD)",
            "ST (PwD)": "ST (PwD)",
            "EWS (PwD)": "EWS (PwD)",
        };
        return map[cat] || cat;
    }

    function formatGender(g) {
        if (g === "Gender-Neutral") return "Male / Any";
        if (g === "Female-only (including Supernumerary)") return "Female";
        return g;
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function animateCounter(elementId, target) {
        const el = document.getElementById(elementId);
        const duration = 600;
        const start = performance.now();
        const initial = parseInt(el.textContent) || 0;

        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(initial + (target - initial) * eased);
            if (progress < 1) requestAnimationFrame(step);
        }

        requestAnimationFrame(step);
    }

    function showLoader(show) {
        loader.classList.toggle("active", show);
        predictBtn.disabled = show;
        predictBtn.textContent = show ? "⏳ Analyzing…" : "🔍 Predict My IITs";
    }

    function hideResults() {
        summarySection.classList.remove("active");
        resultsSection.classList.remove("active");
        noResults.classList.remove("active");
        queryBanner.classList.remove("active");
        resultsBody.innerHTML = "";
    }

    function showError(message) {
        errorToast.textContent = message;
        errorToast.classList.add("visible");
        setTimeout(() => {
            errorToast.classList.remove("visible");
        }, 4000);
    }

})();
