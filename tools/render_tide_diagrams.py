"""Render three presentation diagrams of the implemented TIDE algorithms."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports"/"figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,
                     "pdf.fonttype":42,"font.size":12,"mathtext.fontset":"dejavusans"})
NAVY="#183445";BLUE="#e8f1fc";TEAL="#e2f2eb";ORANGE="#fff0db";GRAY="#f0f3f5"

def sheet(title, subtitle):
    fig,ax=plt.subplots(figsize=(15,11))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.04,.975,title,fontsize=23,fontweight="bold",color=NAVY,va="top")
    ax.text(.04,.925,subtitle,fontsize=12,color="#536572",va="top")
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.02)
    return fig,ax

def box(ax,x,y,w,h,label,fill=BLUE,fs=12):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.008,rounding_size=0.009",facecolor=fill,edgecolor="#a9bbc5",linewidth=1.2))
    ax.text(x,y,label,ha="center",va="center",fontsize=fs,color=NAVY,linespacing=1.65)

def arrow(ax,points,label=None):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color="#54707f",lw=1.5)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=14,color="#54707f",linewidth=1.5))
    if label:
        x,y,text=label
        ax.text(x,y,text,fontsize=11,color="#526674",ha="center",bbox={"facecolor":"white","edgecolor":"none","pad":1})

def original():
    f,a=sheet("TIDE · 从可行步长评价方向","原版使用同状态的三个提案；E006 在全部原试步失败后增加共同缩步。")
    x=.29
    specs=[(.83,.10,"当前参数、共享优化器状态、约束上限\n"+r"$h_i=\tau_i-C_i(\theta)$"),
           (.675,.11,"三个候选，共享旧动量\n"+r"$g^{(k)}=g_F+\lambda_k\sum_i g_i,\quad\lambda_k\in\{0,1,100\}$"),
           (.51,.11,"按需残差 JVP → 方向曲率\n"+r"$q_{ik}=\|J_i d_k\|^2/n_i,\quad p_k=-g_F^\top d_k$"),
           (.34,.115,"选取可兑现收益最大的候选\n"+r"$S_k=p_k\alpha_k,\quad\alpha_k a_{ik}+\alpha_k^2q_{ik}\leq h_i$"),
           (.175,.10,"真实试步复核\n训练 PDE 下降，保护点 IC / BC 均达标")]
    for y,h,t in specs:box(a,x,y,.49,h,t)
    for top,bottom in zip(specs,specs[1:]):
        arrow(a,[(x,top[0]-top[1]/2-.008),(x,bottom[0]+bottom[1]/2+.008)])
    box(a,.79,.51,.30,.12,"上界提前排除无望候选\n"+r"$U_k=p_k\alpha_k^{lin}\geq S_k$"+"\n首选失败后重新开放竞争",GRAY,11)
    arrow(a,[(.535,.51),(.632,.51)])
    box(a,.79,.30,.30,.10,"通过 → 提交参数\n和该候选对应的优化器状态",TEAL)
    arrow(a,[(.535,.175),(.59,.175),(.59,.30),(.632,.30)],(.59,.225,"通过"))
    box(a,.79,.115,.30,.13,"未通过 → 换候选\n全部失败 → 原版零步\n缩步版减半后重新竞争",ORANGE,12)
    arrow(a,[(.535,.15),(.59,.15),(.59,.115),(.632,.115)],(.58,.08,"失败"))
    a.text(.04,.02,"接受后从新参数重新计算；缓存只在当前参数、点集和方向不变时复用。",color="#536572",fontsize=11)
    return f

def twod():
    f,a=sheet("TIDE-2D · 失败原因决定修正平面","E008 实际流程 · 两组共用平面、GN 步长与真实回溯，仅选角评分不同。")
    box(a,.29,.835,.47,.10,"PDE Adam 原提案与待提交状态\n"+r"$(d_0,s_F^+)=Propose(\theta,s,g_F)$")
    box(a,.29,.68,.47,.10,"真实试走原提案\n检查训练 PDE 与两组保护点")
    arrow(a,[(.29,.777),(.29,.738)])
    box(a,.79,.835,.30,.10,"全部通过\n直接提交，零 JVP",TEAL)
    arrow(a,[(.533,.70),(.59,.70),(.59,.835),(.632,.835)],(.58,.77,"通过"))
    box(a,.79,.64,.30,.12,"约束均通过、PDE 未下降\n沿原提案逐次减半\n修正轴退化也回到这条路径",GRAY,11)
    arrow(a,[(.533,.655),(.632,.655)])
    box(a,.29,.51,.47,.12,"约束越界 → 选相对超限最大块\n在原参数处取向内梯度并正交化\n"+r"$D=[d_0,d_1],\quad d_0^\top d_1=0$")
    arrow(a,[(.29,.622),(.29,.578)],(.38,.60,"有约束越界"))
    box(a,.29,.335,.47,.12,"两列 JVP → 完整 2×2 Gram\n"+r"$Z_i=[J_i d_0,J_i d_1],\quad G_i=Z_i^\top Z_i/n_i$"+"\n64 个角度共享同一局部模型")
    arrow(a,[(.29,.442),(.29,.403)])
    box(a,.29,.16,.47,.12,"选一个角、采用该角 GN 步长\n"+r"$k^*=\arg\max_k p_k\alpha_k$"+"\n真实复核，失败时沿该角减半至12次")
    arrow(a,[(.29,.267),(.29,.228)])
    box(a,.79,.335,.30,.12,"真实接受条件\n"+r"$F(\theta+d)<F(\theta)$"+"\n小探针与密集复核均满足容差",TEAL)
    arrow(a,[(.533,.16),(.59,.16),(.59,.335),(.632,.335)])
    arrow(a,[(.79,.572),(.79,.403)])
    box(a,.79,.13,.30,.12,"通过 → 提交修正参数与原提案状态\n全部失败 → 保留原状态，记录停滞",ORANGE,11)
    arrow(a,[(.79,.267),(.79,.198)])
    a.text(.04,.025,"实际实现选中角后仅沿该角回溯。其余角度的真实试算是单独的事后诊断。",color="#536572",fontsize=11)
    return f

def reuse():
    f,a=sheet("TIDE-2D · 一份局部模型，多次复用","每个保护点集沿两根轴各计算一次联合 JVP；小探针与密集复核合计四次。")
    box(a,.185,.78,.28,.17,"当前参数 θ\n残差、PDE 梯度、容差\n基础轴 "+r"$d_0$"+" ＋ 修正轴 "+r"$d_1$",BLUE)
    box(a,.535,.78,.30,.17,"两列残差 JVP\n"+r"$Z_i=[J_i d_0,\ J_i d_1]$"+"\n同时返回各 IC / BC 块",TEAL)
    arrow(a,[(.333,.78),(.377,.78)])
    box(a,.855,.78,.25,.17,"保留向量与交叉项\n"+r"$G_i=Z_i^\top Z_i/n_i$"+"\n"+r"$a_i=2Z_i^\top r_i/n_i$",TEAL)
    arrow(a,[(.693,.78),(.722,.78)])
    box(a,.30,.50,.49,.18,"二维保护模型\n"+r"$\widehat{\Delta C_i}(c)=a_i^\top c+c^\top G_i c$"+"\n"+r"$G_{i,12}=\langle J_i d_0,J_i d_1\rangle/n_i$",BLUE,14)
    arrow(a,[(.855,.687),(.855,.62),(.30,.62),(.30,.598)])
    box(a,.79,.50,.32,.18,"64 个候选角度\n只需二维点积和二次型\n求可行步长、评分及选角\n新组合无需额外 JVP",ORANGE,12)
    arrow(a,[(.553,.50),(.622,.50)])
    box(a,.30,.22,.49,.18,"同一几何，隔离评分贡献\n线性选角 → 共用 GN 步长\n二阶选角 → 共用 GN 步长\n两组均做相同真实复核与回溯",GRAY,13)
    arrow(a,[(.79,.402),(.79,.35),(.30,.35),(.30,.318)])
    box(a,.79,.22,.32,.18,"实际位移\n"+r"$d=\alpha D w,\quad \|w\|=1$"+"\n"+r"$0\leq\alpha\leq1$"+"\n两轴等长，步幅不超过原提案",TEAL,12)
    arrow(a,[(.553,.22),(.622,.22)])
    a.text(.04,.045,"复用范围为本次更新。参数、基底或点集变化后，重新建立局部模型。",color="#536572",fontsize=12)
    return f

if __name__=="__main__":
    with PdfPages(ROOT/"reports"/"TIDE算法流程图.pdf") as pdf:
        for name,fn in (("tide_original",original),("tide_2d",twod),("tide_reuse",reuse)):
            fig=fn()
            fig.savefig(OUT/(name+".png"),dpi=180,facecolor="white")
            pdf.savefig(fig,facecolor="white")
            plt.close(fig)
    print("Rendered 3 PNG diagrams and a 3-page PDF")
