"""Sơ đồ kiến trúc kappa v2 đối chiếu source; không chạy model."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Ellipse
OUT=Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,ax=plt.subplots(figsize=(18,10));ax.set_xlim(0,18);ax.set_ylim(0,10);ax.axis('off')
BLUE='#e1edfa';GREEN='#d9f0e5';ORANGE='#ffe6d5'
def box(x,y,w,h,text,color=BLUE,size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035',facecolor=color,edgecolor='#536172',linewidth=1.3))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(a,b,text=None):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.7,color='#536172',mutation_scale=15))
 if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.15,text,ha='center',fontsize=9)
ax.text(9,9.7,'KAPPA MDN v2: từ quỹ đạo quá khứ đến phân phối GMM tương lai',ha='center',fontsize=19,weight='bold')
box(.2,7.3,2.4,1.4,'Quỹ đạo quá khứ X\n[B, 32, 4]')
box(3.2,7.3,2.2,1.4,'LSTM\n32 trạng thái\n[B, 32, 8]')
box(6,7.3,2.1,1.4,'Lấy trạng thái cuối\nh₃₂: [B, 8]')
box(8.8,7.3,2.5,1.4,'Lớp fc → reshape\n48 × 16 × 6 số\n[B, 48, 96]')
box(12,7.3,5.5,1.4,'6 khối tham số cho mỗi bước tương lai\nμx | μy | log σx | log σy | ρ_pre | z\nMỗi khối chứa 16 số')
for a,b in [(2.6,3.2),(5.4,6),(8.1,8.8),(11.3,12)]:arrow((a,8),(b,8))
box(.2,4.7,3.4,1.6,'THÊM: 48 kappa_logit aₜ\nκₜ = 1 + 15·sigmoid(aₜ)\nκ trong [1,16]; init = 12\nChung mọi mẫu, riêng mỗi bước',GREEN)
box(4.2,4.7,3.7,1.6,'THÊM: cổng mềm khi TRAIN\ngₜ,j = sigmoid((κₜ − j + 0.5)/τ)\nτ: 1 → 0.1 theo lịch đặt trước',GREEN)
arrow((3.6,5.5),(4.2,5.5))
box(8.6,4.7,3.4,1.6,'Giải mã hình dạng Gaussian\nμ = (μx, μy)\nσ = exp(log σ); ρ = tanh(ρ_pre)\nDựng covariance Σ',BLUE,size=10)
box(12.7,4.7,4.8,1.6,'Điểm số z: [B, 48, 16]\nTạo trọng số Gaussian\nnhánh TRAIN hoặc EVAL',ORANGE)
arrow((14.5,7.3),(10.3,6.3),'5 khối hình dạng');arrow((16,7.3),(15.1,6.3),'khối z')
box(.2,1.9,5,1.6,'TRAIN — mục tiêu thay thế\nb = softmax(z); w = b × g\nKHÔNG chuẩn hóa lại: Σⱼ wⱼ = Z ≤ 1\nLoss fit = −log Σⱼ wⱼ Nⱼ(y)\n+ λ × số cổng mềm trung bình',ORANGE,size=11)
arrow((6.05,4.7),(2.7,3.5));ax.text(4.9,4.35,'cổng g',fontsize=9,bbox=dict(facecolor='white',edgecolor='none'));arrow((12.7,5),(5.2,3.1),'điểm số z')
box(6,1.9,5,1.6,'EVAL / TEST — THÊM cổng cứng\nK(t) = clamp(round(κₜ), 1, 16)\nGiữ z của slot 1…K(t)\nSlot còn lại: z = −1e9 → pi = 0\nSoftmax các slot mở: Σⱼ piⱼ = 1',GREEN,size=11)
arrow((3.4,4.7),(6.7,3.5));ax.text(3.3,4.1,'κ đã học',fontsize=9,bbox=dict(facecolor='white',edgecolor='none'));arrow((14.2,4.7),(10.3,3.5),'điểm số z')
box(12,1.65,5.5,2.1,'GMM 2D chuẩn hóa tại mỗi bước t\np(yₜ|X) = Σⱼ piₜ,j N₂(yₜ; μₜ,j, Σₜ,j)\n\n48 phân phối vị trí tương lai\nMỗi bước có tối đa 16 Gaussian\nSlot đóng không đóng góp vào tổng',BLUE,size=11)
arrow((11,2.7),(12,2.7),'pi');arrow((10.3,4.7),(12.6,3.75),'μ, Σ')
ax.text(9,.9,'Màu xanh dương: luồng LSTM–MDN và Gaussian • Xanh lá: cơ chế kappa bổ sung • Cam: trọng số / loss train v2',ha='center',fontsize=11)
ax.text(9,.4,'fc vẫn tính đủ 16 slot. Train dùng tổng mật độ chưa chuẩn hóa; GMM chuẩn hóa bên phải là đầu ra nhánh eval/test.',ha='center',fontsize=11,color='#536172')
fig.savefig(OUT/'17_lstm_to_gmm_kappa_v2.png',dpi=170,bbox_inches='tight',pad_inches=.25);plt.close(fig)
print('Saved architecture figure')
