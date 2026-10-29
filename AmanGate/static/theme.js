// Light / dark theme for every AmanGate page.
// Loaded in <head> (not deferred) so the saved theme is applied before first paint.
// Any element with [data-theme-toggle] switches the theme when clicked.
(function () {
    const KEY = "amangate.theme";
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

    function savedTheme() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }

    function currentTheme() {
        return savedTheme() || (systemDark.matches ? "dark" : "light");
    }

    function syncToggles(theme) {
        const dark = theme === "dark";
        document.querySelectorAll("[data-theme-toggle]").forEach(btn => {
            btn.setAttribute("aria-pressed", String(dark));
            btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
            btn.title = dark ? "Light mode" : "Dark mode";
        });
    }

    function applyTheme(theme) {
        document.documentElement.dataset.theme = theme;
        syncToggles(theme);
    }

    applyTheme(currentTheme());
    document.addEventListener("DOMContentLoaded", () => syncToggles(currentTheme()));

    document.addEventListener("click", (e) => {
        if (!e.target.closest("[data-theme-toggle]")) return;
        const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        try { localStorage.setItem(KEY, next); } catch (err) { /* not saved, still applied */ }
        applyTheme(next);
    });

    // Follow the system setting until the user picks a theme themselves
    systemDark.addEventListener?.("change", (e) => {
        if (!savedTheme()) applyTheme(e.matches ? "dark" : "light");
    });
})();
