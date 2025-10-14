import { bootstrapApplication } from "@angular/platform-browser"
import { provideRouter } from "@angular/router"
import { provideHttpClient, withInterceptorsFromDi } from "@angular/common/http"
import { AppComponent } from "./app/app.component"
import { WidgetEmbedComponent } from "./app/components/widget-embed.component"
import { routes } from "./app/app.routes"
import { provideAnimationsAsync } from "@angular/platform-browser/animations/async"
import { provideServiceWorker } from "@angular/service-worker"
import { HTTP_INTERCEPTORS } from "@angular/common/http"
import { AuthInterceptor } from "./app/interceptors/auth.interceptor"

const ADMIN_PORT = "4300"
const currentPath = window.location.pathname || "/"
if (currentPath === "/" || currentPath === "") {
  if (window.location.port === ADMIN_PORT) {
    // If served on admin port, land on admin area
    window.history.replaceState(null, "", "/admin")
  } else {
    // Otherwise, land on end-user area
    window.history.replaceState(null, "", "/user")
  }
}

const isWidget = window.location.pathname.includes("/widget")

if (isWidget) {
  const host = document.createElement("app-widget-embed")
  document.body.appendChild(host)
} else {
  const host = document.createElement("app-root")
  document.body.appendChild(host)
}

bootstrapApplication(isWidget ? WidgetEmbedComponent : AppComponent, {
  providers: [
    provideHttpClient(withInterceptorsFromDi()),
    provideRouter(routes),
    provideAnimationsAsync(),
    {
      provide: HTTP_INTERCEPTORS,
      useClass: AuthInterceptor,
      multi: true,
    },
    provideServiceWorker("ngsw-worker.js", {
      enabled: false,
      registrationStrategy: "registerWhenStable:30000",
    }),
  ],
})
