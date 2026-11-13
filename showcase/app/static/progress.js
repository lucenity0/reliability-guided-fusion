"use strict";
/* Optional reading-progress rail: a hairline under the top bar that fills as you
   read, and the current § highlighted in the contents. Remove the <script> tag in
   index.html to switch it off; nothing else depends on this file. */
(() => {
  const bar = document.querySelector(".bar");
  if (!bar) return;
  const line = document.createElement("div");
  line.setAttribute("aria-hidden", "true");
  line.style.cssText = "position:absolute;left:0;bottom:-1px;height:2px;width:0;background:var(--ink);transition:width .12s linear";
  bar.appendChild(line);
  const links = [...document.querySelectorAll("#toc a")];
  const sections = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  let raf = 0;
  const update = () => {
    raf = 0;
    const max = document.documentElement.scrollHeight - innerHeight;
    line.style.width = `${max > 0 ? (scrollY / max) * 100 : 0}%`;
    let cur = -1;
    sections.forEach((s, i) => { if (s.getBoundingClientRect().top < innerHeight * 0.35) cur = i; });
    links.forEach((a, i) => a.classList.toggle("on", i === cur));
  };
  addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(update); }, { passive: true });
  addEventListener("resize", update);
  update();
})();
