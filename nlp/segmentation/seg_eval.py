#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
 
if __name__=="__main__":
    pred_word_count = 0
    target_word_count = 0
    correct_word_count = 0
    for line in sys.stdin:
        fields = line.strip().decode('utf8').split()
        if len(fields) != 4:
            continue
    
        target, pred = fields[2:4]
        if pred in ('E', 'S'):
            pred_word_count += 1
            if target == pred:
                correct_word_count +=1
    
        if target in ('E', 'S'):
            target_word_count += 1
 
    P = correct_word_count / float(pred_word_count)
    R = correct_word_count / float(target_word_count)
    F1 = (2 * P * R) / (P + R)
    
    print('  --> Word count of predict, golden and correct : %d, %d, %d' %
            (pred_word_count, target_word_count, correct_word_count))
    print("  --> P = %f, R = %f, F1 = %f" % (P, R, F1))

