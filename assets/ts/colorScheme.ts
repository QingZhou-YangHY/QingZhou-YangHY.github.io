/**
 * Project override of the theme's color-scheme controller.
 *
 * The stock version cycles through a 3-state model (light / dark / auto) and
 * collapses back to `auto` whenever the chosen scheme matches the OS, which
 * makes the toggle feel unreliable. This is a clean 2-state switch: every click
 * flips light <-> dark and persists exactly what the user picked. First-time
 * visitors are seeded from the OS preference, after which the choice is explicit.
 */
type colorScheme = 'light' | 'dark';

class StackColorScheme {
    private localStorageKey = 'StackColorScheme';
    private currentScheme: colorScheme;

    constructor(toggleEl: HTMLElement) {
        this.currentScheme = this.getSavedScheme();
        this.setBodyClass();

        if (toggleEl) this.bindClick(toggleEl);

        if (document.body.style.transition == '')
            document.body.style.setProperty('transition', 'background-color .3s ease');
    }

    private saveScheme() {
        localStorage.setItem(this.localStorageKey, this.currentScheme);
    }

    private bindClick(toggleEl: HTMLElement) {
        toggleEl.addEventListener('click', () => {
            this.currentScheme = this.currentScheme === 'dark' ? 'light' : 'dark';
            this.setBodyClass();
            this.saveScheme();
        });
    }

    private setBodyClass() {
        document.documentElement.dataset.scheme = this.currentScheme;
        window.dispatchEvent(
            new CustomEvent('onColorSchemeChange', { detail: this.currentScheme })
        );
    }

    private getSavedScheme(): colorScheme {
        const saved = localStorage.getItem(this.localStorageKey);
        if (saved === 'dark') return 'dark';
        if (saved === 'light') return 'light';
        // First visit (or legacy 'auto'): seed from OS once, then it's explicit.
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
}

export default StackColorScheme;
