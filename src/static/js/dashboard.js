async function toggleTrading() {
    const toggleButton = document.getElementById('toggle-trading');
    const currentState = toggleButton.textContent.toLowerCase().includes('enable') ? 'enable' : 'disable';
    
    try {
        // Disable button during request
        toggleButton.disabled = true;
        
        // Try the trading-data toggle endpoint first
        try {
            const response = await fetch('/api/trading-data/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                toggleButton.textContent = currentState === 'enable' ? 'Disable Trading' : 'Enable Trading';
                showNotification('success', data.message);
                // Update the trading data display if needed
                if (typeof updateTradingData === 'function' && data.data) {
                    updateTradingData(data.data);
                }
                return;
            } else {
                throw new Error(data.message || 'Failed to toggle trading status');
            }
        } catch (error) {
            console.warn('Trading data toggle failed, falling back to simple toggle:', error);
            
            // Fallback to simple enable/disable endpoint
            const response = await fetch(`/api/${currentState}-trading`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                toggleButton.textContent = currentState === 'enable' ? 'Disable Trading' : 'Enable Trading';
                showNotification('success', data.message);
            } else {
                throw new Error(data.message || 'Failed to toggle trading status');
            }
        }
    } catch (error) {
        console.error('Error toggling trading status:', error);
        showNotification('error', error.message || 'Failed to toggle trading status');
    } finally {
        toggleButton.disabled = false;
    }
} 