"use client"

import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts"
import { useRecentActivity } from "@/lib/api"
import { Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useEffect, useState } from "react"

// Fallback mock data in case the API endpoint is not available
const MOCK_ACTIVITY_DATA = {
  todayTrades: 54,
  weeklyData: [
    {"day": "Mon", "trades": 12, "volume": 1650},
    {"day": "Tue", "trades": 10, "volume": 1450},
    {"day": "Wed", "trades": 15, "volume": 2100},
    {"day": "Thu", "trades": 18, "volume": 2500},
    {"day": "Fri", "trades": 22, "volume": 3300},
    {"day": "Sat", "trades": 30, "volume": 5000},
    {"day": "Sun", "trades": 25, "volume": 3500}
  ]
};

export function Overview() {
  const { activity: apiActivity, isLoading, error } = useRecentActivity();
  const [activity, setActivity] = useState(MOCK_ACTIVITY_DATA);
  
  // Use API data if available, otherwise use mock data
  useEffect(() => {
    if (apiActivity && !error) {
      setActivity(apiActivity);
    } else if (error && !isLoading) {
      console.log("Using mock activity data due to API error");
      setActivity(MOCK_ACTIVITY_DATA);
    }
  }, [apiActivity, error, isLoading]);

  if (isLoading) {
    return (
      <div className="flex h-[350px] w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-lg">Loading activity data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-[350px] w-full items-center justify-center">
        <p className="text-destructive">Error loading activity data</p>
      </div>
    );
  }

  if (!activity || !activity.weeklyData || activity.weeklyData.length === 0) {
    return (
      <div className="flex h-[350px] w-full items-center justify-center">
        <p className="text-muted-foreground">No activity data available</p>
      </div>
    );
  }

  return (
    <div className="h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={activity.weeklyData}>
          <XAxis dataKey="day" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis
            stroke="#888888"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `$${value}`}
          />
          <Tooltip
            formatter={(value) => [`$${value}`, "Volume"]}
            labelFormatter={(label) => `${label}`}
          />
          <Bar dataKey="volume" name="Trading Volume" fill="currentColor" radius={[4, 4, 0, 0]} className="fill-primary" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

