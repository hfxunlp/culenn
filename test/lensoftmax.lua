require "cunn"
require "culenn"
tmodstd=nn.SoftMax():cuda()
tmod=lenn.LenSoftMax():cuda()
minbsize=4
maxbsize=4
minlen=5
maxlen=8
minpadlen=3
maxpadlen=3
psg=true
firstcycle=5
for t=1, firstcycle do
	if psg then
		bsize=math.random(minbsize, maxbsize)
		lens=math.random(minlen, maxlen)
		plens=math.random(minpadlen, maxpadlen)
		lvec=torch.LongTensor(bsize):fill(lens):cudaLong()
		stdi=torch.randn(bsize, lens):cuda()
		i=torch.cat(stdi, torch.randn(bsize, plens):cuda())
		stdgo=torch.randn(bsize, lens):cuda()
		go=torch.cat(stdgo, torch.randn(bsize, plens):cuda())
		stdo=tmodstd:forward(stdi)
		o=tmod:forward({i, lvec})
		if not (o:narrow(2, 1, lens):equal(stdo) and o:narrow(2, lens+1, plens):equal(torch.zeros(bsize, plens):cuda()) ) then
			psg=false
			print("forward error")
		end
		stdgi=tmodstd:backward(stdi, stdgo)
		gi=tmod:backward({i, lvec}, go)[1]
		if not (gi:narrow(2, 1, lens):equal(stdgi) and gi:narrow(2, lens+1, plens):equal(torch.zeros(bsize, plens):cuda()) ) then
			psg=false
			print("backward error")
		end
	end
end
if psg then
	print("test pass")
end
