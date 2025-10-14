import { Injectable } from "@angular/core"
import { BehaviorSubject } from "rxjs"

export type Theme = "light" | "dark" | "system"

@Injectable({
  providedIn: "root",
})
export class ThemeService {
  private themeKey = "app_theme"
  private themeSubject = new BehaviorSubject<Theme>(this.getInitialTheme())
  public theme$ = this.themeSubject.asObservable()

  constructor() {
    this.applyTheme(this.themeSubject.value)
    this.listenForSystemChanges()
  }

  private getInitialTheme(): Theme {
    const savedTheme = localStorage.getItem(this.themeKey) as Theme
    if (savedTheme && ["light", "dark", "system"].includes(savedTheme)) {
      return savedTheme
    }
    return "system"
  }

  setTheme(theme: Theme): void {
    localStorage.setItem(this.themeKey, theme)
    this.themeSubject.next(theme)
    this.applyTheme(theme)
  }

  private applyTheme(theme: Theme): void {
    let effectiveTheme: "light" | "dark"

    if (theme === "system") {
      effectiveTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
    } else {
      effectiveTheme = theme
    }

    document.documentElement.classList.remove("light-theme", "dark-theme")
    document.documentElement.classList.add(`${effectiveTheme}-theme`)

    // For Angular Material
    if (effectiveTheme === "dark") {
      document.body.classList.add("dark-theme")
    } else {
      document.body.classList.remove("dark-theme")
    }
  }

  private listenForSystemChanges(): void {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
      if (this.themeSubject.value === "system") {
        this.applyTheme("system")
      }
    })
  }
}
