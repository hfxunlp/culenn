require "cunn"
require "culenn"
tmodstd=nn.SoftMax():cuda()
tmod=lenn.LenSoftMax():cuda()
tmodstd1=nn.SoftMax():cuda()
minbsize=8
maxbsize=64
minlen=16
maxlen=128
minpadlen=8
maxpadlen=32
psg=true
firstcycle=100
for t=1, firstcycle do
	if psg then
		bsize=math.random(minbsize, maxbsize)
		lens=math.random(minlen, maxlen)
		plens=math.random(minpadlen, maxpadlen)
		eplens=math.random(lens+1,plens+lens-1)
		lvec=torch.LongTensor(bsize):fill(lens):cudaLong()
		lvec[bsize]=eplens
		stdi=torch.randn(bsize, lens):cuda()
		i=torch.cat(stdi, torch.randn(bsize, plens):cuda())
		stdgo=torch.randn(bsize, lens):cuda()
		go=torch.cat(stdgo, torch.randn(bsize, plens):cuda())
		stdo=tmodstd:forward(stdi:narrow(1,1,bsize-1))
		stdo1=tmodstd1:forward(i[-1]:narrow(1, 1, eplens))
		o=tmod:forward({i, lvec})
		if not (o[-1]:narrow(1, 1, eplens):equal(stdo1) and o[-1]:narrow(1, eplens+1, plens+lens-eplens):equal(torch.zeros(plens+lens-eplens):cuda()) and o:narrow(1,1,bsize-1):narrow(2, 1, lens):equal(stdo) and o:narrow(1,1,bsize-1):narrow(2, lens+1, plens):equal(torch.zeros(bsize-1, plens):cuda())) then
			psg=false
			print("forward error")
			print(stdo)
			print(stdo1)
			print(o)
		end
		stdgi=tmodstd:backward(stdi:narrow(1,1,bsize-1), stdgo:narrow(1,1,bsize-1))
		stdgi1=tmodstd1:backward(i[-1]:narrow(1, 1, eplens), go[-1]:narrow(1, 1, eplens))
		gi=tmod:backward({i, lvec}, go)[1]
		if not (gi[-1]:narrow(1, 1, eplens):equal(stdgi1) and gi[-1]:narrow(1, eplens+1, plens+lens-eplens):equal(torch.zeros(plens+lens-eplens):cuda()) and gi:narrow(1,1,bsize-1):narrow(2, 1, lens):equal(stdgi) and gi:narrow(1,1,bsize-1):narrow(2, lens+1, plens):equal(torch.zeros(bsize-1, plens):cuda())) then
			psg=false
			print("backward error")
			print(stdgi)
			print(stdgi1)
			print(gi)
		end
		xlua.progress(t, firstcycle)
	end
end
if psg then
	print("test pass")
end
