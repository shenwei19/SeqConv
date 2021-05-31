library(motifStack)

name <- commandArgs(T)[1]
file <- paste0('motif/',name,'_seq.meme')
mat <- read.table(file)
mat <- t(mat)
rownames(mat) <- c('A','C','G','T')
motif <- new("pcm", mat=as.matrix(mat),name='test')

pdf_name <- paste0('motif/',name,'_motif.pdf')
pdf(pdf_name)
plot(motif)
dev.off()



