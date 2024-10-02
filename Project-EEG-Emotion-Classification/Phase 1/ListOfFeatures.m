function [output] = ListOfFeatures(input)
    if (input == 1)
        output = 'Theta';
    end

    if (input == 2)
        output = 'Alpha';
    end

    if (input == 3)
        output = 'LBeta';
    end

    if (input == 4)
        output = 'MBeta';
    end

    if (input == 5)
        output = 'HBeta';
    end

    if (input == 6)
        output = 'Gamma';
    end

    if (input == 7)
        output = 'Median Frequency';
    end

    if (input == 8)
        output = 'Mean Frequency';
    end

    if (input == 9)
        output = 'Variance';
    end

    if (input == 10)
        output = 'Maximum Abs';
    end

    if (input == 11)
        output = 'Kurtosis';
    end

    if (input == 12)
        output = '99 Percent Bandwidth';
    end

    if (input == 13)
        output = 'Maximum Power Frequency';
    end

    if (input == 14)
        output = 'Band Power Feature';
    end

end

