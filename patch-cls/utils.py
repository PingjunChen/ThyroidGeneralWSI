from visdom import Visdom


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, epoch, train_acc, val_acc, win, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 2)).cpu() * epoch,
        Y=torch.Tensor([train_acc, val_acc]).unsqueeze(0).cpu() / epoch_size,
        win=win,
        update=update_type
    )
