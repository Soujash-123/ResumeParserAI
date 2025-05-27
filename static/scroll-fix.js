// Add this script to your index.html page
document.addEventListener('DOMContentLoaded', function() {
    // Get the match button element
    const matchBtn = document.getElementById('match-btn');
    
    // Add a click event listener
    if (matchBtn) {
        const originalClickHandler = matchBtn.onclick;
        
        matchBtn.onclick = function(event) {
            // Call the original handler if it exists
            if (typeof originalClickHandler === 'function') {
                originalClickHandler.call(this, event);
            }
            
            // Wait for the results to be processed and displayed
            setTimeout(function() {
                // Scroll to the results section
                const resultSection = document.querySelector('.result-section');
                if (resultSection) {
                    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 1000); // Wait 1 second for the results to be displayed
        };
    }
    
    // Also add scrolling for the parse and process buttons
    const parseBtn = document.getElementById('parse-btn');
    const processBtn = document.getElementById('process-btn');
    
    if (parseBtn) {
        parseBtn.addEventListener('click', function() {
            setTimeout(function() {
                const resultSection = document.querySelector('.result-section');
                if (resultSection) {
                    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 1000);
        });
    }
    
    if (processBtn) {
        processBtn.addEventListener('click', function() {
            setTimeout(function() {
                const resultSection = document.querySelector('.result-section');
                if (resultSection) {
                    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 1000);
        });
    }
});
