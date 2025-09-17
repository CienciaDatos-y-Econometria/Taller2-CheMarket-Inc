#-----------------------------------------
#PART 2
#
#
#
#
#-----------------------------------------

# Libraries

require(pacman)
p_load(dplyr, tidyr, tidyverse, readr, ggplot2, corrplot)

#Change data name
db2 <- parte_b
##Validation of treatment-control group


#Validation by easier signup
t.test(time_spent ~ easier_signup, data = db2)

#Validation by past sesion
t.test(past_sessions ~ easier_signup, data = db2)

#Validation by device_type
table(db2$easier_signup, db2$device_type) %>% chisq.test()

#Validation by os_type
table(db2$easier_signup, db2$os_type) %>% chisq.test()

#Validation by sign_up
table(db2$easier_signup, db2$sign_up) %>% chisq.test()

#model sign_up~easier_signup
model_2 <- lm(sign_up ~ easier_signup,
            data = db2,)
summary(model_2)

#model revenue~sign_up
model1_2 <- lm(Revenue~sign_up,data = db2)
summary(model1_2)




  



