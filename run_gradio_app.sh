#!/bin/bash

# Real-Time Interactive Video Generation with Gradio
echo "🎬 Starting Real-Time Video Generation with Gradio..."

# Install Gradio if needed
if ! python -c "import gradio" &> /dev/null; then
    echo "📦 Installing Gradio..."
    pip install gradio
fi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1

# Launch the Gradio app
echo "🚀 Launching Gradio app..."
echo ""
echo "Features:"
echo "  ✨ Much better real-time streaming than Streamlit"  
echo "  🔄 Built-in auto-refresh every 500ms"
echo "  ⏱️ Generates for exactly 1 minute"
echo "  📺 Native video frame handling"
echo "  🎛️ Clean, responsive interface"
echo ""
echo "🔗 App will be available at:"
echo "   Local: http://localhost:7860"
echo "   SSH Tunnel: ssh -L 7860:localhost:7860 user@host"
echo ""

python gradio_realtime_app.py