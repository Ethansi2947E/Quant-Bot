"use client"

import { useEffect, useState } from "react"
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Button } from "@/components/ui/button"
import { usePerformanceMetrics } from "@/lib/api"
import { Loader2 } from "lucide-react"

// Add interface for props
interface ProfitChartProps {
  onTimeframeChange?: (timeframe: string) => void;
}

export function ProfitChart({ onTimeframeChange }: ProfitChartProps) {
  // Use these exact timeframe names to match the API
  const [activeTimeframe, setActiveTimeframe] = useState("daily");
  const { metrics, isLoading, refetch } = usePerformanceMetrics();
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    if (metrics && metrics.chartData) {
      // Select the appropriate timeframe data based on active timeframe
      if (activeTimeframe === "daily" && metrics.chartData.daily) {
        setData(formatChartData(metrics.chartData.daily));
      } else if (activeTimeframe === "weekly" && metrics.chartData.weekly) {
        setData(formatChartData(metrics.chartData.weekly));
      } else if (activeTimeframe === "monthly" && metrics.chartData.monthly) {
        setData(formatChartData(metrics.chartData.monthly));
      }
    }
  }, [metrics, activeTimeframe]);

  // Format data for the chart
  const formatChartData = (chartData: any[]) => {
    if (!chartData || !Array.isArray(chartData)) {
      console.warn("Invalid chart data received:", chartData);
      return [];
    }
    return chartData.map(item => ({
      date: formatDate(item.timestamp, activeTimeframe),
      profit: item.pnl,
      value: item.value
    }));
  };

  // Format date based on timeframe
  const formatDate = (dateStr: string, timeframeType: string) => {
    if (!dateStr) return "";
    try {
      const date = new Date(dateStr);
      if (timeframeType === "daily") {
        return date.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' });
      } else if (timeframeType === "weekly") {
        return `Week ${Math.ceil(date.getDate() / 7)}`;
      } else {
        return date.toLocaleDateString('en-US', { month: 'short' });
      }
    } catch (e) {
      console.error("Error formatting date:", e);
      return dateStr;
    }
  };

  const handleTimeframeChange = (tf: string) => {
    console.log(`Switching to timeframe: ${tf}`);
    setActiveTimeframe(tf);
    
    // Map frontend timeframe names to API timeframe format
    let apiTimeframe = '1D';
    if (tf === 'weekly') apiTimeframe = '1W';
    if (tf === 'monthly') apiTimeframe = '1M';
    
    // Call parent component's handler if provided
    if (onTimeframeChange) {
      onTimeframeChange(apiTimeframe);
    }
    
    // Trigger refetch
    refetch();
  };

  const formatCurrency = (value: number) => {
    return `$${value.toFixed(2)}`
  };

  // Calculate min and max values for Y axis
  const getMinMaxValues = () => {
    if (!data || data.length === 0) return { min: -500, max: 500 };
    
    const values = data.map(item => item.profit);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    // Add some padding
    const padding = Math.max(200, (max - min) * 0.2);
    return { 
      min: Math.floor(min - padding), 
      max: Math.ceil(max + padding) 
    };
  };

  const { min, max } = getMinMaxValues();

  if (isLoading) {
    return (
      <div className="flex h-[300px] w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-lg">Loading chart data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end">
        <div className="flex items-center space-x-2">
          <Button
            variant={activeTimeframe === "daily" ? "default" : "outline"}
            size="sm"
            onClick={() => handleTimeframeChange("daily")}
          >
            Daily
          </Button>
          <Button
            variant={activeTimeframe === "weekly" ? "default" : "outline"}
            size="sm"
            onClick={() => handleTimeframeChange("weekly")}
          >
            Weekly
          </Button>
          <Button
            variant={activeTimeframe === "monthly" ? "default" : "outline"}
            size="sm"
            onClick={() => handleTimeframeChange("monthly")}
          >
            Monthly
          </Button>
        </div>
      </div>
      <div className="h-[300px] w-full">
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis 
                domain={[min, max]}
                tickFormatter={formatCurrency} 
                tick={{ fontSize: 12 }} 
                tickLine={false} 
                axisLine={false} 
              />
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <Tooltip
                formatter={(value: number) => [`${value >= 0 ? "+" : ""}$${Math.abs(value).toFixed(2)}`, "Profit/Loss"]}
              />
              <Area
                type="monotone"
                dataKey="profit"
                stroke="#10b981"
                fill="url(#colorProfit)"
                strokeWidth={2}
                activeDot={{ r: 6 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <p className="text-muted-foreground">No profit data available</p>
          </div>
        )}
      </div>
    </div>
  )
}

