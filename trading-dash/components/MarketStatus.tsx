'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Clock, Globe } from 'lucide-react'

function getMarketStatus() {
  const now = new Date()
  const hour = now.getHours()
  const day = now.getDay()
  
  // Weekend check - Market is closed on Saturday
  if (day === 6) {
    return {
      isOpen: false,
      session: 'Closed (Weekend)',
      nextOpen: 'Sunday 11 PM'
    }
  }

  // Sunday check - Market opens at 11 PM
  if (day === 0) {
    if (hour < 23) {
      return {
        isOpen: false,
        session: 'Closed (Weekend)',
        nextOpen: 'Today 11 PM'
      }
    }
  }

  // Friday check - Market closes at 10 PM
  if (day === 5 && hour >= 22) {
    return {
      isOpen: false,
      session: 'Closed (Weekend)',
      nextOpen: 'Sunday 11 PM'
    }
  }

  // Market sessions (all in local time)
  if (hour >= 23 || hour < 8) {
    return {
      isOpen: true,
      session: 'Asian Session',
      timeLeft: hour >= 23 ? (8 - (hour - 23)) : (8 - hour)
    }
  } else if (hour >= 8 && hour < 16) {
    return {
      isOpen: true,
      session: 'London Session',
      timeLeft: 16 - hour
    }
  } else if (hour >= 13 && hour < 22) {
    return {
      isOpen: true,
      session: 'New York Session',
      timeLeft: 22 - hour
    }
  }

  return {
    isOpen: true,
    session: 'Market Open',
    timeLeft: 24 - hour
  }
}

export function MarketStatus() {
  const [status, setStatus] = useState(getMarketStatus())
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => {
      setStatus(getMarketStatus())
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  return (
    <Card className="bg-white/5 backdrop-blur-xl border-white/10">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-indigo-500/10 flex items-center justify-center">
              <Globe className="h-5 w-5 text-indigo-500" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <p className="text-sm font-medium text-white">Market Status</p>
                <span className={`h-2 w-2 rounded-full ${status.isOpen ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
              </div>
              <p className="text-xs text-white/60 mt-0.5">{status.session}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-orange-500/10 flex items-center justify-center">
              <Clock className="h-5 w-5 text-orange-500" />
            </div>
            <div>
              <p className="text-sm font-medium text-white">
                {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </p>
              <p className="text-xs text-white/60 mt-0.5">
                {status.isOpen 
                  ? `${status.timeLeft}h until session end` 
                  : `Opens ${status.nextOpen}`}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 