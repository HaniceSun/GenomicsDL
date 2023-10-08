### https://martha-labbook.netlify.app/posts/extracting-data-for-variants-common-in-both-file-sets/
import subprocess
import os
import sys

def CombineDataSets(inF1, inF2):
    inF1x = inF1.split('/')[-1]
    inF2x = inF2.split('/')[-1]
    ouF = '%s_%s_OverlappedSnps.txt'%(inF1x, inF2x)
    ouFa = inF1x + '_ExtractedWith_' + inF2x
    ouFb = inF2x + '_ExtractedWith_' + inF1x
    ouFc = inF2x + '_' + inF1x + '_MergeList.txt'
    ouFd = inF2x + '_' + inF1x + '_OverlappedMerged'
    ouFe = inF2x + '_' + inF1x + '_MultiPosMisSnps.txt'

    ouFile = open(ouF, 'w')
    D = {}
    inFile = open(inF2 + '.bim')
    for line in inFile:
        fields = line.split()
        D[fields[1]] = True
    inFile.close()

    inFile = open(inF1 + '.bim')
    for line in inFile:
        fields = line.split()
        if fields[1] in D:
            ouFile.write(fields[1] + '\n')
    ouFile.close()

    subprocess.call("plink --bfile %s --keep-allele-order --extract %s --make-bed --out %s"%(inF1, ouF, ouFa), shell=True)
    subprocess.call("plink --bfile %s --keep-allele-order --extract %s --make-bed --out %s"%(inF2, ouF, ouFb), shell=True)

    ouFile = open(ouFc, 'w')
    ouFile.write(ouFa + '\n')
    ouFile.write(ouFb + '\n')
    ouFile.close()

    subprocess.call("plink --merge-list %s --keep-allele-order --make-bed --out %s"%(ouFc, ouFd), shell=True)

    if not os.path.exists(ouFd + '.bed'):
        ouFile = open(ouFe, 'w')
        inFile = open(ouFd + '.log')
        for line in inFile:
            line = line.strip()
            if line.find('Warning: Multiple positions seen for variant') == 0:
                snp = line.split("'")[1]
                ouFile.write(snp + '\n')
        inFile.close()

        inFile = open(ouFd + '-merge.missnp')
        for line in inFile:
            line = line.strip()
            ouFile.write(line + '\n')
        inFile.close()
        ouFile.close()

        subprocess.call("plink --bfile %s --keep-allele-order --extract %s --exclude %s --make-bed --out %s"%(inF1, ouF, ouFe, ouFa), shell=True)
        subprocess.call("plink --bfile %s --keep-allele-order --extract %s --exclude %s --make-bed --out %s"%(inF2, ouF, ouFe, ouFb), shell=True)
        subprocess.call("plink --merge-list %s --keep-allele-order --make-bed --out %s"%(ouFc, ouFd), shell=True)

    subprocess.call("plink --merge-list %s --keep-allele-order --snps-only just-acgt --make-bed --out %s"%(ouFc, ouFd), shell=True)


CombineDataSets(sys.argv[1], sys.argv[2])
