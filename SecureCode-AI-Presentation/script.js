document.addEventListener('DOMContentLoaded', () => {
    // --- EXISTING SLIDE NAVIGATION LOGIC ---
    const slidesWrapper = document.getElementById('slides-wrapper');
    const slides = document.querySelectorAll('.slide');
    const chapterItems = document.querySelectorAll('.chapter-item');

    let currentSlideIndex = 0;
    const totalSlides = slides.length;
    let isScrolling = false;

    // Initialize UI
    updateUI();

    // Sidebar click navigation
    chapterItems.forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.getAttribute('data-slide'));
            goToSlide(index);
        });
    });

    // Wheel Scroll (Debounced)
    window.addEventListener('wheel', (e) => {
        if (isScrolling) return;
        if (e.deltaY > 0) {
            nextSlide();
        } else {
            prevSlide();
        }
    }, { passive: false });

    // Keyboard Navigation
    window.addEventListener('keydown', (e) => {
        const key = e.key;
        if (['ArrowDown', 'ArrowRight', ' ', 'PageDown'].includes(key)) {
            e.preventDefault();
            nextSlide();
        } else if (['ArrowUp', 'ArrowLeft', 'PageUp'].includes(key)) {
            e.preventDefault();
            prevSlide();
        } else if (key === 'Home') {
            e.preventDefault();
            goToSlide(0);
        } else if (key === 'End') {
            e.preventDefault();
            goToSlide(totalSlides - 1);
        }
    });

    // Touch support
    let touchStartY = 0;
    window.addEventListener('touchstart', (e) => {
        touchStartY = e.touches[0].clientY;
    }, { passive: true });

    window.addEventListener('touchend', (e) => {
        const touchEndY = e.changedTouches[0].clientY;
        const diff = touchStartY - touchEndY;

        if (Math.abs(diff) > 50) {
            if (diff > 0) nextSlide();
            else prevSlide();
        }
    }, { passive: true });

    function nextSlide() {
        if (currentSlideIndex < totalSlides - 1) {
            goToSlide(currentSlideIndex + 1);
        }
    }

    function prevSlide() {
        if (currentSlideIndex > 0) {
            goToSlide(currentSlideIndex - 1);
        }
    }

    function goToSlide(index) {
        if (index < 0 || index >= totalSlides || isScrolling) return;

        isScrolling = true;
        currentSlideIndex = index;
        updateUI();

        setTimeout(() => {
            isScrolling = false;
        }, 800);
    }

    function updateUI() {
        slidesWrapper.style.transform = `translateY(-${currentSlideIndex * 100}vh)`;

        chapterItems.forEach(item => {
            item.classList.remove('active');
        });

        const activeItem = document.querySelector(`.chapter-item[data-slide="${currentSlideIndex}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
        }

        // Re-trigger animations
        const currentSlideEl = slides[currentSlideIndex];
        const animatedElements = currentSlideEl.querySelectorAll('.animate-in');

        animatedElements.forEach(el => {
            el.style.animation = 'none';
            el.offsetHeight; /* Trigger reflow */
            el.style.animation = '';
        });
    }

    window.addEventListener('resize', () => {
        slidesWrapper.style.transform = `translateY(-${currentSlideIndex * 100}vh)`;
    });

    // --- NEW: FLUID ANIMATION (DEEP COLORS) ---
    initFluidAnimation();
});

function initFluidAnimation() {
    const canvas = document.getElementById('neuro-bg');
    const ctx = canvas.getContext('2d');

    let w, h;
    let orbs = [];

    // UPDATED CONFIG: Deeper, darker colors
    const config = {
        count: 20,           // More orbs for denser coverage
        minSize: 300,
        maxSize: 800,
        speed: 0.3,          // Slower, more majestic
        colors: [
            '#1a0b2e', // Deep Indigo
            '#2d0055', // Dark Violet
            '#004a55', // Deep Teal/Cyan
            '#4a004a', // Dark Magenta
            '#020024'  // Midnight Blue
        ]
    };

    window.addEventListener('resize', resize);
    resize();

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }

    function hexToRgb(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    class Orb {
        constructor() {
            this.init();
        }

        init() {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
            this.vx = (Math.random() - 0.5) * config.speed;
            this.vy = (Math.random() - 0.5) * config.speed;
            this.size = Math.random() * (config.maxSize - config.minSize) + config.minSize;

            const hex = config.colors[Math.floor(Math.random() * config.colors.length)];
            this.rgb = hexToRgb(hex);

            // Lower opacity for deeper blending without washout
            this.maxOpacity = Math.random() * 0.3 + 0.1;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            const buffer = this.size / 2;
            if (this.x < -buffer || this.x > w + buffer) this.vx *= -1;
            if (this.y < -buffer || this.y > h + buffer) this.vy *= -1;
        }

        draw() {
            const g = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.size);
            g.addColorStop(0, `rgba(${this.rgb.r}, ${this.rgb.g}, ${this.rgb.b}, ${this.maxOpacity})`);
            g.addColorStop(1, `rgba(${this.rgb.r}, ${this.rgb.g}, ${this.rgb.b}, 0)`);

            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function initOrbs() {
        orbs = [];
        for (let i = 0; i < config.count; i++) {
            orbs.push(new Orb());
        }
        animate();
    }

    function animate() {
        requestAnimationFrame(animate);
        ctx.clearRect(0, 0, w, h);

        // "Lighten" blend mode works well for deep colors on black
        ctx.globalCompositeOperation = 'lighten';

        for (let i = 0; i < orbs.length; i++) {
            orbs[i].update();
            orbs[i].draw();
        }

        ctx.globalCompositeOperation = 'source-over';
    }

    initOrbs();
}