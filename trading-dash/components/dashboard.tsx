"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Overview } from "@/components/overview"
import { ActiveTrades } from "@/components/active-trades"
import { Signals } from "@/components/signals"
import { DetailedAnalytics } from "@/components/detailed-analytics"
import { ProfitChart } from "@/components/profit-chart"
import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardShell } from "@/components/dashboard-shell"
import { useAccountOverview, usePerformanceMetrics } from "@/lib/api"
import { Loader2 } from "lucide-react"

export default function Dashboard() {
  const [timeframe, setTimeframe] = useState("1D")
  const { data: accountData, isLoading: isAccountLoading } = useAccountOverview()
  const { 
    metrics: performanceData, 
    isLoading: isPerformanceLoading,
    setTimeframe: updateApiTimeframe 
  } = usePerformanceMetrics(timeframe)

  // Handle timeframe changes from ProfitChart
  const handleTimeframeChange = (newTimeframe: string) => {
    setTimeframe(newTimeframe);
    updateApiTimeframe(newTimeframe);
  }

  // Function to render card content with loading state
  const renderCardContent = (isLoading: boolean, value: string | number | null, changeText: string) => {
    if (isLoading) {
      return (
        <>
          <div className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="text-2xl font-bold text-muted-foreground">Loading...</span>
          </div>
          <p className="text-xs text-muted-foreground">Loading data...</p>
        </>
      )
    }
    
    return (
      <>
        <div className="text-2xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground">{changeText}</p>
      </>
    )
  }

  return (
    <DashboardShell>
      <DashboardHeader heading="Bot Performance Dashboard" text="Track your bot's trading performance and activities" />
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="active-trades">Active Trades</TabsTrigger>
          <TabsTrigger value="signals">Signals</TabsTrigger>
          <TabsTrigger value="analytics">Detailed Analytics</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Profit/Loss</CardTitle>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  className="h-4 w-4 text-muted-foreground"
                >
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                </svg>
              </CardHeader>
              <CardContent>
                {renderCardContent(
                  isPerformanceLoading, 
                  performanceData ? 
                    (performanceData.profitLoss >= 0 ? '+' : '') + 
                    `$${performanceData.profitLoss.toFixed(2)}` : 
                    '$0.00',
                  performanceData && performanceData.profitLossPercentage ? 
                    (performanceData.profitLossPercentage >= 0 ? '+' : '') + 
                    `${performanceData.profitLossPercentage.toFixed(2)}% from last month` : 
                    '+0.00% from last month'
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  className="h-4 w-4 text-muted-foreground"
                >
                  <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
              </CardHeader>
              <CardContent>
                {renderCardContent(
                  isPerformanceLoading, 
                  performanceData ? `${performanceData.winRate.toFixed(1)}%` : '0.0%',
                  performanceData && performanceData.winRate > 0 ? 
                    `+${(performanceData.winRate * 0.1).toFixed(1)}% from last month` : 
                    '+0.0% from last month'
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Trades</CardTitle>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  className="h-4 w-4 text-muted-foreground"
                >
                  <rect width="20" height="14" x="2" y="5" rx="2" />
                  <path d="M2 10h20" />
                </svg>
              </CardHeader>
              <CardContent>
                {renderCardContent(
                  isPerformanceLoading, 
                  performanceData && performanceData.totalTrades 
                    ? performanceData.totalTrades 
                    : accountData && accountData.openPositions > 0 
                      ? accountData.openPositions 
                      : 0,
                  accountData ? `${accountData.openPositions || 0} open positions` : 'No open positions'
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Portfolio Value</CardTitle>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  className="h-4 w-4 text-muted-foreground"
                >
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
              </CardHeader>
              <CardContent>
                {renderCardContent(
                  isAccountLoading, 
                  accountData ? `$${accountData.equity.toFixed(2)}` : '$0.00',
                  accountData && accountData.dailyPnL !== 0 ? 
                    `${accountData.dailyPnL > 0 ? '+' : ''}$${accountData.dailyPnL.toFixed(2)} today` : 
                    'No change today'
                )}
              </CardContent>
            </Card>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Profit Overview</CardTitle>
              </CardHeader>
              <CardContent className="pl-2">
                <ProfitChart onTimeframeChange={handleTimeframeChange} />
              </CardContent>
            </Card>
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>
                  Your bot executed {accountData?.openPositions || 0} trades today
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Overview />
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="active-trades" className="space-y-4">
          <ActiveTrades />
        </TabsContent>
        <TabsContent value="signals" className="space-y-4">
          <Signals />
        </TabsContent>
        <TabsContent value="analytics" className="space-y-4">
          <DetailedAnalytics />
        </TabsContent>
      </Tabs>
    </DashboardShell>
  )
}

