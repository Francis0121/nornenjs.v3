package com.nornenjs.web.actor;

import com.nornenjs.web.group.Authority;
import com.nornenjs.web.group.GroupMembers;
import com.nornenjs.web.group.GroupService;
import com.nornenjs.web.util.Pagination;
import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.support.MessageSourceAccessor;
import org.springframework.security.authentication.dao.SaltSource;
import org.springframework.security.authentication.encoding.PasswordEncoder;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.AuthorityUtils;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by Francis on 2015-04-23.
 */
@Service
public class ActorServiceImpl extends SqlSessionDaoSupport implements ActorService, UserDetailsService{

    @Autowired
    private MessageSourceAccessor messages;
    @Autowired
    private GroupService groupService;
    @Autowired
    private PasswordEncoder passwordEncoder;
    @Autowired
    private SaltSource saltSource;
    

    // ~ Actor service
    @Override
    public Actor selectOne(Integer pn) {
        return getSqlSession().selectOne("actor.selectOne", pn);
    }

    @Override
    public Integer selectCount(ActorFilter actorFilter) {
        return getSqlSession().selectOne("selectCount", actorFilter);
    }

    @Override
    public List<Actor> selectList(ActorFilter actorFilter) {
        Pagination pagination = actorFilter.getPagination();
        Integer count = selectCount(actorFilter);
        pagination.setNumItems(count);
        if(count == 0){
            return new ArrayList<Actor>();
        }
        return getSqlSession().selectList("selectList", actorFilter);
    }

    @Override
    public Integer insert(Actor actor) {
        setEncodedPassword(actor);
        return getSqlSession().insert("actor.insert", actor);
    }

    @Override
    public Integer update(Actor actor) {
        return getSqlSession().update("actor.update", actor);
    }

    @Override
    public Integer delete(Integer pn) {
        return getSqlSession().delete("actor.delete", pn);
    }
    
    @Override
    public Boolean selectUsernameExist(String username) {
        // TODO Make test case
        Integer count = getSqlSession().selectOne("actor.selectUsernameExist", username);
        return count > 0;
    }

    @Override
    public Boolean selectEmailExist(String email) {
        // TODO Make test case
        Integer count = getSqlSession().selectOne("actor.selectEmailExist", email);
        return count > 0;
    }

    @Override
    public Boolean createActor(ActorInfo actorInfo) {
        // TODO Make test case
        // 로그인 DB 입력
        Actor actor = actorInfo.getActor();
        actor.setEnabled(true);
        Integer result = insert(actor);
        if(!result.equals(1)) return false;
        // 사용자 정보 입력
        result = getSqlSession().insert("actor.insertActorInfo", actorInfo);
        if(!result.equals(1)) return false;
        
        // 그룹 입력
        result = groupService.insertGroupMembers(new GroupMembers(actor.getUsername(), Authority.DOCTOR.getValue()));
        return result.equals(1);
    }

    @Override
    public String selectUsernameFromEmail(String username) {
        return getSqlSession().selectOne("actor.selectUsernameFromEmail", username);
    }

    @Override
    public String selectEmailFromUsername(String email) {
        return getSqlSession().selectOne("actor.selectEmailFromUsername", email);
    }

    @Override
    public ActorInfo selectActorInfoFromUsername(String username) {
        return getSqlSession().selectOne("actor.selectActorInfoFromUsername", username);
    }

    @Override
    public Integer updateActorInfo(ActorInfo actorInfo) {
        return getSqlSession().update("actor.updateActorInfo", actorInfo);
    }

    @Override
    public Boolean updatePassword(Actor actor) {
        Integer result = getSqlSession().update("actor.updatePassword", actor);
        if(!result.equals(1)) return false;
        
        result = getSqlSession().update("actor.updateDate", actor);
        return result.equals(1);
    }

    @Override
    public Boolean selectIsRightPassword(Actor actor) {
        logger.debug(actor.toString());
        actor.setEnabled(true);
        setEncodedPassword(actor);
        Actor getActor = getSqlSession().selectOne("actor.selectOneFromUsername", actor.getUsername());
        
        logger.debug(actor.toString());
        logger.debug(getActor.toString());
        return actor.getPassword().equals(getActor.getPassword());
    }

    // ~ UserDetailService

    public void setEncodedPassword(Actor actor) {
        User user = new User(actor.getUsername(), actor.getPassword(), actor.getEnabled(), true, true, true, AuthorityUtils.NO_AUTHORITIES);
        String encodedPassword = passwordEncoder.encodePassword(actor.getPassword(), saltSource.getSalt(user));
        actor.setPassword(encodedPassword);
    }
    
    /**
     * Allows subclasses to add their own granted authorities to the list to be returned in the <tt>UserDetails</tt>.
     *
     * @param username the username, for use by finder methods
     * @param authorities the current granted authorities, as populated from the <code>authoritiesByUsername</code>
     *        mapping
     */
    protected void addCustomAuthorities(String username, List<GrantedAuthority> authorities) {}
    
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        List<Actor> actors = selectUsersByUsername(username);
        
        if(actors.size() == 0){
            logger.debug("Query returned no results for user '" + username + "'");
            
            //TODO Need to Change JdbcDaoImpl.notFound
            throw new UsernameNotFoundException(messages.getMessage("JdbcDaoImpl.notFound", new Object[]{username}, "Username {0} not found"));
        }
        
        // ~ select user information
        Actor actor = actors.get(0);
        UserDetails user = new User(actor.getUsername(), actor.getPassword(), actor.getEnabled(), true, true, true, AuthorityUtils.NO_AUTHORITIES);
        
        // ~ Set authority
        Set<GrantedAuthority> dbAuthsSet = new HashSet<GrantedAuthority>();
        dbAuthsSet.addAll(groupService.selectGroupAuthoritiesInfo(user.getUsername()));

        List<GrantedAuthority> dbAuths = new ArrayList<GrantedAuthority>(dbAuthsSet);
        addCustomAuthorities(user.getUsername(), dbAuths);

        if (dbAuths.size() == 0) {
            logger.debug("User '" + username + "' has no authorities and will be treated as 'not found'");

            //TODO Need to Change JdbcDaoImpl.noAuthority
            throw new UsernameNotFoundException(messages.getMessage("JdbcDaoImpl.noAuthority", new Object[] {username}, "User {0} has no GrantedAuthority"));
        }
        
        return createUserDetails(username, user, dbAuths);
    }

    private List<Actor> selectUsersByUsername(String username) {
        List<Actor> actors = getSqlSession().selectList("actor.selectUsersByUsername", username);
        return actors;
    }

    protected UserDetails createUserDetails(String username, UserDetails userFromUserQuery,
                                            List<GrantedAuthority> combinedAuthorities) {
        logger.debug(combinedAuthorities);
        return new User(username, userFromUserQuery.getPassword(), userFromUserQuery.isEnabled(),
                true, true, true, combinedAuthorities);
    }
    
}
