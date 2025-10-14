import { Injectable } from "@angular/core"
import { BehaviorSubject } from "rxjs"

export interface NotificationMessage {
  id: string
  message: string
  type: "info" | "success" | "warning" | "error"
  timestamp: Date
  read: boolean
}

@Injectable({
  providedIn: "root",
})
export class SupportService {
  private unreadCountSubject = new BehaviorSubject<number>(0)
  private notificationsSubject = new BehaviorSubject<NotificationMessage[]>([])

  public unreadCount$ = this.unreadCountSubject.asObservable()
  public notifications$ = this.notificationsSubject.asObservable()

  constructor() {
    this.addNotification({
      id: "1",
      message: "Welcome! Click here if you need any help getting started.",
      type: "info",
      timestamp: new Date(),
      read: false,
    })
  }

  addNotification(notification: NotificationMessage) {
    const current = this.notificationsSubject.value
    this.notificationsSubject.next([notification, ...current])
    this.updateUnreadCount()
  }

  markAsRead(notificationId: string) {
    const current = this.notificationsSubject.value
    const updated = current.map((n) => (n.id === notificationId ? { ...n, read: true } : n))
    this.notificationsSubject.next(updated)
    this.updateUnreadCount()
  }

  markAllAsRead() {
    const current = this.notificationsSubject.value
    const updated = current.map((n) => ({ ...n, read: true }))
    this.notificationsSubject.next(updated)
    this.updateUnreadCount()
  }

  private updateUnreadCount() {
    const unread = this.notificationsSubject.value.filter((n) => !n.read).length
    this.unreadCountSubject.next(unread)
  }

  clearNotifications() {
    this.notificationsSubject.next([])
    this.unreadCountSubject.next(0)
  }
}
