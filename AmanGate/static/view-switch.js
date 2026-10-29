// Operations ⇄ Analytics switch in the header.
// Browsers with cross-page view transitions slide the whole page (see admin.css /
// system.css). Others get a quick slide of the dark highlight before the page changes.
document.addEventListener("click", (e) => {
    const link = e.target.closest(".ag-view-switch a:not([aria-current])");
    if (!link || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
    if ("PageRevealEvent" in window) return;  // the view transition animates it

    e.preventDefault();
    const sw = link.closest(".ag-view-switch");
    sw.classList.toggle("is-right");
    sw.querySelector("[aria-current]")?.removeAttribute("aria-current");
    link.setAttribute("aria-current", "page");
    setTimeout(() => { location.href = link.href; }, 240);
});
