// Application Data
const appData = {
  documents: [
    {
      id: "doc_1", 
      title: "Q3 Marketing Campaign Results",
      content: "Email marketing campaign showed 15% conversion rate in treatment group vs 12% in control. Customer segments with high CLV responded best to personalized offers.",
      category: "marketing",
      confidence: 0.89
    },
    {
      id: "doc_2",
      title: "Product Launch Strategy",
      content: "New product features increased user engagement by 23% among early adopters. Premium tier customers showed 2.3x higher adoption rates.",
      category: "product",
      confidence: 0.92
    },
    {
      id: "doc_3", 
      title: "Customer Retention Analysis",
      content: "Loyalty program reduced churn by 18% overall. Effect was heterogeneous - millennials showed 25% churn reduction while Gen X showed 12%.",
      category: "retention",
      confidence: 0.87
    }
  ],
  experiments: [
    {
      name: "Email Personalization",
      treatment_effect: 0.15,
      control_rate: 0.12,
      treatment_rate: 0.27,
      confidence_interval: [0.08, 0.22],
      segments: [
        {"name": "High CLV", "uplift": 0.22, "size": 2500},
        {"name": "Medium CLV", "uplift": 0.15, "size": 5000},
        {"name": "Low CLV", "uplift": 0.08, "size": 7500}
      ]
    },
    {
      name: "Discount Offers",
      treatment_effect: 0.12,
      control_rate: 0.18,
      treatment_rate: 0.30,
      confidence_interval: [0.06, 0.18],
      segments: [
        {"name": "Price Sensitive", "uplift": 0.19, "size": 3000},
        {"name": "Brand Loyal", "uplift": 0.09, "size": 4000},
        {"name": "Occasional Buyers", "uplift": 0.14, "size": 8000}
      ]
    }
  ],
  qini_data: [
    {"percentile": 10, "qini_score": 0.08},
    {"percentile": 20, "qini_score": 0.15},
    {"percentile": 30, "qini_score": 0.21},
    {"percentile": 40, "qini_score": 0.26},
    {"percentile": 50, "qini_score": 0.29},
    {"percentile": 60, "qini_score": 0.31},
    {"percentile": 70, "qini_score": 0.32},
    {"percentile": 80, "qini_score": 0.32},
    {"percentile": 90, "qini_score": 0.31},
    {"percentile": 100, "qini_score": 0.30}
  ],
  feature_importance: [
    {"feature": "Customer Lifetime Value", "importance": 0.35},
    {"feature": "Previous Purchase Frequency", "importance": 0.28},
    {"feature": "Age Group", "importance": 0.18},
    {"feature": "Geographic Region", "importance": 0.12},
    {"feature": "Channel Preference", "importance": 0.07}
  ],
  business_metrics: {
    projected_revenue_lift: 125000,
    cost_per_treatment: 2.50,
    roi_estimate: 2.8,
    confidence_level: 0.95
  }
};

// Chart color palette
const chartColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'];

// Global chart instances
let charts = {};

// DOM Elements
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const searchResults = document.getElementById('search-results');
const suggestionChips = document.querySelectorAll('.suggestion-chip');
const loadingOverlay = document.getElementById('loading-overlay');

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
  initializeTabNavigation();
  initializeSearch();
  initializeCharts();
  setupEventListeners();
});

// Tab Navigation
function initializeTabNavigation() {
  tabButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      const targetTab = e.target.dataset.tab;
      switchTab(targetTab);
    });
  });
}

function switchTab(targetTab) {
  // Update tab buttons
  tabButtons.forEach(btn => btn.classList.remove('active'));
  document.querySelector(`[data-tab="${targetTab}"]`).classList.add('active');
  
  // Update tab content
  tabContents.forEach(content => content.classList.remove('active'));
  document.getElementById(targetTab).classList.add('active');
  
  // Initialize charts if switching to uplift modeling
  if (targetTab === 'uplift-modeling') {
    setTimeout(() => {
      initializeCharts();
    }, 100);
  }
}

// Search Functionality
function initializeSearch() {
  searchButton.addEventListener('click', performSearch);
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      performSearch();
    }
  });
  
  suggestionChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      searchInput.value = e.target.textContent;
      performSearch();
    });
  });
}

function performSearch() {
  const query = searchInput.value.trim();
  if (!query) return;
  
  showLoading();
  
  // Simulate API delay
  setTimeout(() => {
    const results = searchDocuments(query);
    displaySearchResults(results);
    hideLoading();
  }, 1200);
}

function searchDocuments(query) {
  const queryLower = query.toLowerCase();
  const results = [];
  
  appData.documents.forEach(doc => {
    const contentLower = doc.content.toLowerCase();
    const titleLower = doc.title.toLowerCase();
    
    // Simple relevance scoring
    let relevanceScore = 0;
    let highlightedContent = doc.content;
    
    const queryWords = queryLower.split(' ');
    queryWords.forEach(word => {
      if (word.length > 2) {
        if (titleLower.includes(word)) relevanceScore += 0.3;
        if (contentLower.includes(word)) {
          relevanceScore += 0.1;
          // Highlight matching words
          const regex = new RegExp(`(${word})`, 'gi');
          highlightedContent = highlightedContent.replace(regex, '<span class="result-highlight">$1</span>');
        }
      }
    });
    
    if (relevanceScore > 0) {
      results.push({
        ...doc,
        relevanceScore,
        highlightedContent
      });
    }
  });
  
  return results.sort((a, b) => b.relevanceScore - a.relevanceScore);
}

function displaySearchResults(results) {
  if (results.length === 0) {
    searchResults.innerHTML = `
      <div class="search-result">
        <div class="result-header">
          <h3 class="result-title">No results found</h3>
        </div>
        <p class="result-content">Try different search terms or check our document suggestions.</p>
      </div>
    `;
    return;
  }
  
  const resultsHTML = results.map(result => `
    <div class="search-result">
      <div class="result-header">
        <h3 class="result-title">${result.title}</h3>
        <span class="confidence-score">${Math.round(result.confidence * 100)}% confidence</span>
      </div>
      <div class="result-content">
        ${result.highlightedContent}
      </div>
      <div class="result-metadata">
        <span>Category: ${result.category}</span>
        <span>Relevance: ${Math.round(result.relevanceScore * 100)}%</span>
      </div>
    </div>
  `).join('');
  
  searchResults.innerHTML = resultsHTML;
}

// Chart Initialization
function initializeCharts() {
  // Destroy existing charts
  Object.values(charts).forEach(chart => {
    if (chart) chart.destroy();
  });
  charts = {};
  
  createUpliftChart();
  createQiniChart();
  createFeatureChart();
  createExperimentChart();
}

function createUpliftChart() {
  const ctx = document.getElementById('uplift-chart');
  if (!ctx) return;
  
  const segments = appData.experiments[0].segments;
  
  charts.uplift = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: segments.map(s => s.name),
      datasets: [{
        label: 'Uplift %',
        data: segments.map(s => s.uplift * 100),
        backgroundColor: chartColors.slice(0, segments.length),
        borderColor: chartColors.slice(0, segments.length),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `Uplift: ${context.parsed.y.toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Uplift Percentage'
          }
        }
      }
    }
  });
}

function createQiniChart() {
  const ctx = document.getElementById('qini-chart');
  if (!ctx) return;
  
  charts.qini = new Chart(ctx, {
    type: 'line',
    data: {
      labels: appData.qini_data.map(d => `${d.percentile}%`),
      datasets: [{
        label: 'Qini Score',
        data: appData.qini_data.map(d => d.qini_score),
        borderColor: chartColors[0],
        backgroundColor: chartColors[0] + '20',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Population Percentile'
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Qini Score'
          }
        }
      }
    }
  });
}

function createFeatureChart() {
  const ctx = document.getElementById('feature-chart');
  if (!ctx) return;
  
  charts.feature = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: appData.feature_importance.map(f => f.feature),
      datasets: [{
        data: appData.feature_importance.map(f => f.importance * 100),
        backgroundColor: chartColors.slice(0, appData.feature_importance.length),
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            usePointStyle: true,
            padding: 15
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.label}: ${context.parsed.toFixed(1)}%`;
            }
          }
        }
      }
    }
  });
}

function createExperimentChart() {
  const ctx = document.getElementById('experiment-chart');
  if (!ctx) return;
  
  charts.experiment = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: appData.experiments.map(e => e.name),
      datasets: [
        {
          label: 'Control Rate',
          data: appData.experiments.map(e => e.control_rate * 100),
          backgroundColor: chartColors[3],
          borderColor: chartColors[3],
          borderWidth: 1
        },
        {
          label: 'Treatment Rate',
          data: appData.experiments.map(e => e.treatment_rate * 100),
          backgroundColor: chartColors[0],
          borderColor: chartColors[0],
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Conversion Rate (%)'
          }
        }
      }
    }
  });
}

// Utility Functions
function showLoading() {
  loadingOverlay.classList.add('active');
}

function hideLoading() {
  loadingOverlay.classList.remove('active');
}

function setupEventListeners() {
  // Add hover effects for interactive elements
  document.querySelectorAll('.overview-card, .recommendation-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-4px)';
    });
    
    card.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
    });
  });
  
  // Handle window resize for charts
  window.addEventListener('resize', debounce(() => {
    Object.values(charts).forEach(chart => {
      if (chart) {
        chart.resize();
      }
    });
  }, 250));
}

// Debounce function for performance
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Mock real-time updates (simulate live data)
function startRealTimeUpdates() {
  setInterval(() => {
    // Simulate small fluctuations in metrics
    const statusIndicators = document.querySelectorAll('.status-dot');
    statusIndicators.forEach(dot => {
      dot.style.opacity = Math.random() > 0.5 ? '1' : '0.5';
    });
  }, 3000);
}

// Initialize real-time updates
setTimeout(startRealTimeUpdates, 2000);

// Export for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    searchDocuments,
    appData,
    chartColors
  };
}