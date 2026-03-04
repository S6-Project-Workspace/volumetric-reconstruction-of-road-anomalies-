// Professional Presentation Navigation Script

let currentSlide = 0;
const slides = document.querySelectorAll('.slide');
const navDots = document.querySelectorAll('.nav-dot');
const totalSlides = slides.length;

// Initialize
function init() {
    showSlide(0);
    setupEventListeners();
}

// Show specific slide
function showSlide(index) {
    // Ensure index is within bounds
    if (index < 0) index = 0;
    if (index >= totalSlides) index = totalSlides - 1;
    
    // Hide all slides
    slides.forEach(slide => {
        slide.classList.remove('active');
    });
    
    // Show current slide
    slides[index].classList.add('active');
    currentSlide = index;
    
    // Update navigation dots
    updateNavDots();
    
    // Update URL hash
    window.location.hash = index;
}

// Update navigation dots
function updateNavDots() {
    navDots.forEach((dot, index) => {
        if (index === currentSlide) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
}

// Navigate to next slide
function nextSlide() {
    if (currentSlide < totalSlides - 1) {
        showSlide(currentSlide + 1);
    }
}

// Navigate to previous slide
function prevSlide() {
    if (currentSlide > 0) {
        showSlide(currentSlide - 1);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        switch(e.key) {
            case 'ArrowRight':
            case 'ArrowDown':
            case 'PageDown':
            case ' ': // Spacebar
                e.preventDefault();
                nextSlide();
                break;
            case 'ArrowLeft':
            case 'ArrowUp':
            case 'PageUp':
                e.preventDefault();
                prevSlide();
                break;
            case 'Home':
                e.preventDefault();
                showSlide(0);
                break;
            case 'End':
                e.preventDefault();
                showSlide(totalSlides - 1);
                break;
        }
    });
    
    // Navigation dot clicks
    navDots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            showSlide(index);
        });
    });
    
    // Mouse wheel navigation (optional)
    let wheelTimeout;
    document.addEventListener('wheel', (e) => {
        clearTimeout(wheelTimeout);
        wheelTimeout = setTimeout(() => {
            if (e.deltaY > 0) {
                nextSlide();
            } else if (e.deltaY < 0) {
                prevSlide();
            }
        }, 100);
    }, { passive: true });
    
    // Touch swipe navigation
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });
    
    document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    }, { passive: true });
    
    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                // Swipe left - next slide
                nextSlide();
            } else {
                // Swipe right - previous slide
                prevSlide();
            }
        }
    }
    
    // Handle URL hash on load
    window.addEventListener('load', () => {
        const hash = window.location.hash.substring(1);
        if (hash && !isNaN(hash)) {
            showSlide(parseInt(hash));
        }
    });
    
    // Handle browser back/forward
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.substring(1);
        if (hash && !isNaN(hash)) {
            showSlide(parseInt(hash));
        }
    });
}

// Prevent context menu on presentation
document.addEventListener('contextmenu', (e) => {
    e.preventDefault();
});

// Prevent text selection during presentation
document.addEventListener('selectstart', (e) => {
    if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
    }
});

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Fullscreen toggle (F11 or F key)
document.addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') {
        toggleFullscreen();
    }
});

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            console.log(`Error attempting to enable fullscreen: ${err.message}`);
        });
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

// Slide counter display (optional - can be toggled with 'c' key)
let showCounter = false;
const counterDiv = document.createElement('div');
counterDiv.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 10px 15px;
    border-radius: 5px;
    font-family: monospace;
    font-size: 14px;
    display: none;
    z-index: 1000;
`;
document.body.appendChild(counterDiv);

document.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C') {
        showCounter = !showCounter;
        updateCounter();
    }
});

function updateCounter() {
    if (showCounter) {
        counterDiv.textContent = `${currentSlide + 1} / ${totalSlides}`;
        counterDiv.style.display = 'block';
    } else {
        counterDiv.style.display = 'none';
    }
}

// Update counter when slide changes
const originalShowSlide = showSlide;
showSlide = function(index) {
    originalShowSlide(index);
    updateCounter();
};

// Print mode (Ctrl+P or Cmd+P)
window.addEventListener('beforeprint', () => {
    // Show all slides for printing
    slides.forEach(slide => {
        slide.style.position = 'relative';
        slide.style.opacity = '1';
        slide.style.visibility = 'visible';
        slide.style.pageBreakAfter = 'always';
    });
});

window.addEventListener('afterprint', () => {
    // Restore normal view
    slides.forEach((slide, index) => {
        slide.style.position = 'absolute';
        if (index !== currentSlide) {
            slide.style.opacity = '0';
            slide.style.visibility = 'hidden';
        }
        slide.style.pageBreakAfter = 'auto';
    });
});

console.log('Presentation Controls:');
console.log('→ / ↓ / Space: Next slide');
console.log('← / ↑: Previous slide');
console.log('Home: First slide');
console.log('End: Last slide');
console.log('F: Toggle fullscreen');
console.log('C: Toggle slide counter');
console.log('Mouse wheel: Navigate slides');
