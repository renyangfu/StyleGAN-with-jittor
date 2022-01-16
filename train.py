import argparse
import random
import math
import os
import jittor as jt
from jittor import nn
from jittor import optim, transform

from tqdm import tqdm
import numpy as np
from PIL import Image
jt.flags.use_cuda = 1

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator, requires_grad


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k] * decay + (1 - decay) * par2[k].detach())


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    #loader = dataset.set_attrs(shuffle=False, batch_size=batch_size, num_workers=0, drop_last=True)
    # CHANGE BACK
    loader = dataset.set_attrs(shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)
    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


#import pdb
def train(args, dataset, generator, discriminator, transforms):
    #pdb.set_trace()
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3_000_000))

    # generator.requires_grad_step(step, False)
    # discriminator.requires_grad_step(step, True)
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False
    log_interval = 100

    for i in pbar:
        # discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        ckpt_step = step
        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            #loader = sample_data(
            #    dataset, args.batch.get(resolution, args.batch_default), resolution
            #)
            #data_loader = iter(loader)
            dataset = MultiResolutionDataset(args.path, transforms) ##this
            loader = sample_data(
                    dataset, args.batch.get(resolution, args.batch_default), resolution)
            data_loader = iter(loader)
            #dataset.resolution = resolution
            print("resolution", resolution)
            jt.gc()
            # jt.cudnn.set_max_workspace_ratio(0.0)
            jt.display_memory_info()

            '''
            jt.save(
                {
                    'generator.g': generator.generator.state_dict(),
                    'generator.s': generator.style.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.param_groups,
                    'd_optimizer': d_optimizer.param_groups,
                    'g_running.g': g_running.generator.state_dict(),
                    'g_running.s': g_running.style.state_dict()
                },
                f'checkpoint/train_step-{ckpt_step}.model',
             )
             '''
            generator.save(f'checkpoint/train_step-{ckpt_step}-g.pkl')
            discriminator.save(f'checkpoint/train_step-{ckpt_step}-d.pkl')
            g_running.save(f'checkpoint/train_step-{ckpt_step}-r.pkl')



            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)
            #print("data loader error")
        used_sample += real_image.shape[0]

        b_size = real_image.size(0)

        # real_image.start_grad()
        # print("Before Discrim")
        # real_image = jt.array(np.random.random(real_image.shape).astype("float32"))
        real_image = real_image.clone()
        real_scores = discriminator(real_image, step=step, alpha=alpha)

        real_predict = nn.softplus(-real_scores).mean()
        # real_predict.backward(retain_graph=True)
        # print("After Discrim")

        grad_real = jt.grad(
            real_scores.sum(), real_image
        )
        # print(grad_real)
        grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty
        # grad_penalty.backward()
        if i % log_interval == 0:
            grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.init.gauss((4, b_size, code_size), "float32").chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = jt.init.gauss((2, b_size, code_size,), "float32").chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        fake_predict = nn.softplus(fake_predict).mean()
        # fake_predict.backward()
        if i % log_interval == 0:
            disc_loss_val = (real_predict + fake_predict).item()

        d_loss = real_predict + grad_penalty + fake_predict
        # print(d_loss)
        d_optimizer.step(d_loss)

        # print("After Discrim Loss")
        # jt.ones((3,52,102)).transpose((1,2,0)).sync(True)

        if (i + 1) % n_critic == 0:
            # generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            # generator.requires_grad_step(step, True)
            # discriminator.requires_grad_step(step, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)
            
            # print(predict.min(), predict.max())
            loss = nn.softplus(-predict).mean()
            # loss = nn.softplus((-predict).maximum(0.0)).mean()
            # loss = nn.softplus(-predict).mean()

            # loss.backward()
            g_optimizer.step(loss)

            if i % log_interval == 0:
                gen_loss_val = loss.item()

            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

            # generator.requires_grad_step(step, False)
            # discriminator.requires_grad_step(step, True)

        # jt.sync_all()
            # jt.ones((3,52,102)).transpose((1,2,0)).sync(True)
        if (i + 1) % 100 == 0:
            # jt.display_memory_info()
            # jt.clear_trace_data()
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))
            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            jt.init.gauss((gen_j, code_size), "float32"), step=step, alpha=alpha
                        ).numpy()
                    )
            if not os.path.exists(f'sample/{step}'):
                os.mkdir(f'sample/{step}')

            jt.save_image(
                jt.concat(images, 0),
                f'sample/{step}/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )
            if (i+1) % 100000 == 0:
                generator.save(f'checkpoint/train_step-{ckpt_step-i}-g.pkl')
                discriminator.save(f'checkpoint/train_step-{ckpt_step-i}-d.pkl')
                g_running.save(f'checkpoint/train_step-{ckpt_step-i}-r.pkl')

        # if (i + 1) % 10000 == 0:
        #     torch.save(
        #         g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
        #     )
        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)
        # if i == 3: break
        # break

if __name__ == '__main__':
    code_size = 512
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, default='../StyleGAN/lmdb', help='path of specified dataset', nargs='?')
    parser.add_argument(
        '--phase',
        type=int,
        default=200_000,        # Origin is 600k
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=128, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )

    args = parser.parse_args()

    generator = StyledGenerator(code_size)
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    g_running = StyledGenerator(code_size)
    g_running.eval()

    g_optimizer = optim.Adam(
        [{'params': generator.generator.parameters()},
         {'params': generator.style.parameters(), 'lr': args.lr * 0.01, 'mult': 0.01}], lr=args.lr, betas=(0.0, 0.99)
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    # for i, v in enumerate(discriminator.parameters()):
        # print(i, v.name(), v.shape)

    accumulate(g_running, generator, 0)

    if args.ckpt is not None:
        if os.path.exists(args.ckpt + '-g.pkl') and os.path.exists(args.ckpt + '-d.pkl') \
                and os.path.exists(args.ckpt + '-r.pkl'):
            generator.load(args.ckpt + '-g.pkl')
            discriminator.load(args.ckpt + '-d.pkl')
            g_running.load(args.ckpt+'-r.pkl')


        #ckpt = torch.load(args.ckpt)

        #generator.load_state_dict(ckpt['generator'])
        #discriminator.load_state_dict(ckpt['discriminator'])
        #g_running.load_state_dict(ckpt['g_running'])
        #g_optimizer.load_state_dict(ckpt['g_optimizer'])
        #d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transforms = transform.Compose(
        [
            transform.RandomHorizontalFlip(),
            # transform.ToTensor(),
            transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # from pycg import nas
    # args.path = nas.parse_input_path(args.path, do_sync=True)

    dataset = MultiResolutionDataset(args.path, transforms)
    print(len(dataset))
    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 4 

    # import numpy as np
    # from jittor_utils import auto_diff

    # hook = auto_diff.Hook("generator")
    # hook.hook_module(generator)
    # hook.hook_optimizer(g_optimizer)
    # hook.hook_module(discriminator)
    # hook.hook_optimizer(d_optimizer)
    # breakpoint()

    train(args, dataset, generator, discriminator, transforms)
