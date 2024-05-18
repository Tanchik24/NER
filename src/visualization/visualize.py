import plotly.graph_objects as go


def plot_losses(train_loss, valid_loss):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(train_loss) + 1)),
        y=train_loss,
        mode='lines+markers',
        name='Train Loss'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(1, len(valid_loss) + 1)),
        y=valid_loss,
        mode='lines+markers',
        name='Validation Loss'
    ))

    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0, y=1.0),
        margin=dict(l=40, r=0, t=40, b=30)
    )

    fig.show()