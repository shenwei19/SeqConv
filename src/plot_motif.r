library(motifStack)

file <- commandArgs(T)[1]
name <- commandArgs(T)[2]
#file <- paste0('motif/',name,'_seq.meme')
mat <- read.table(file)
mat <- t(mat)
rownames(mat) <- c('A','C','G','T')
motif <- new("pcm", mat=as.matrix(mat),name=name)

png_name <- paste0(name,'_motif.png')
png(png_name)
plot(motif)
dev.off()



