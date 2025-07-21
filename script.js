const slides = document.querySelectorAll(".phone-intro .intro-slide");
let current = 0;

function nextSlide() {
  slides[current].classList.remove("active");
  slides[current].style.opacity = 0;

  current = (current + 1) % slides.length;

  // Slight delay before showing the next slide
  setTimeout(() => {
    slides[current].classList.add("active");
    slides[current].style.opacity = 1;
  }, 200);
}

// Start slide loop
setInterval(nextSlide, 3000);



